# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os
import re
import cv2
import numpy as np
import subprocess

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

from pprint import pprint
from datetime import datetime

# natural sort
def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "frames")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in natural_sort(os.listdir(image_dir))}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "config.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split(' = ') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    seq_info = gather_sequence_info(sequence_dir, detection_file)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []

    test_dict = {}

    def frame_callback(vis, frame_idx):
        if frame_idx % 100 == 0:
            print("Tracked frame %d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]


        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

            
        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Update visualization.
        if display:
            # print("came here-1")
            image = cv2.imread(
                seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            # cv2.imshow("test", image)
            # cv2.waitKey()
            # print("came here-2")

            vis.set_image(image.copy())
            # print("came here-3")
            
            vis.draw_detections(detections)
            # print("came here-4")
            
            vis.draw_trackers(tracker.tracks, test_dict)
            # print("came here-5")

        new_track_id = []
        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
            # print(frame_idx, track.track_id)

            new_track_id.append(track.track_id)

        # print("track-id-list:", new_track_id)

        for i in new_track_id:
            if not i in test_dict:
                test_dict[i] = datetime.now().strftime('%H:%M:%S')
        
        # print(test_dict)


    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Store results.
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    # parser.add_argument(
    #     "--sequence_dir", help="Path to MOTChallenge sequence directory",
    #     default=None, required=True)
    # parser.add_argument(
    #     "--detection_file", help="Path to custom detections.", default=None,
    #     required=True)
    # parser.add_argument(
    #     "--output_file", help="Path to the tracking output file. This file will"
    #     " contain the tracking results on completion.",
    #     default="/tmp/hypotheses.txt")
    # parser.add_argument(
    #     "--min_confidence", help="Detection confidence threshold. Disregard "
    #     "all detections that have a confidence lower than this value.",
    #     default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    # parser.add_argument(
    #     "--nn_budget", help="Maximum size of the appearance descriptors "
    #     "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=False, type=bool)
    parser.add_argument(
        "--video_name", help="Path to data/video_name sequence directory",
        default=None, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("----------------------------------")
    sequence_dir = "../data/" + args.video_name
    subprocess.call("mkdir -p output && cd output && mkdir -p " + args.video_name, shell=True)
    detection_file = "./resources/detections/" + args.video_name + "/det.npy"
    output_file = "./output/" + args.video_name + "/deep-sort-output.txt"
    min_confidence = 0.5
    nn_budget = 100
    display = True
    run(
        sequence_dir, detection_file, output_file,
        min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, nn_budget, display)
    
    print("----------------------------------")