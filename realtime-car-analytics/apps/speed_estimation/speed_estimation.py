import sys

import numpy as np
import configparser
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import cv2

import applications.roi.speed_estimation.draw_custom_roi_hscd as hscd

frame_rate = 0
distance = 0
speed_threshold = 0

polygon = None
top_polygon = None
bottom_polygon = None

roi_points = []
read_config_file_points = False
output_directory_created = False

real_time_tracked_points = np.zeros((1, 5))


# Function Name: init_set_roi
# Description: Calls define_roi method from draw_custom_roi_hscd module
# Input: (video_path: Path to input video)
# Output: Saves config.ini in path: realtime_car_analytics/data/<video name>/config/
def init_set_roi(video_path):
    """ Call define_roi method """
    hscd.define_roi(video_path)


# Function Name: get_roi_points
# Description: Get roi points from config.ini file and return all the points in a list
# Input: (select_polygon: -1 for bottom polygon | 0 for roi | 1 for top polygon)
# Output: Returns list of polygon points
def get_roi_points(config, select_polygon):
    """ Get roi points from config.ini file and return all the points in a list """
    boundary_points = []

    if select_polygon == -1:
        lower_left = (eval(config['outer_bottom_left']))
        upper_left = (eval(config['inner_bottom_left']))
        upper_right = (eval(config['inner_bottom_right']))
        lower_right = (eval(config['outer_bottom_right']))

    elif select_polygon == 1:
        lower_left = (eval(config['inner_top_left']))
        upper_left = (eval(config['outer_top_left']))
        upper_right = (eval(config['outer_top_right']))
        lower_right = (eval(config['inner_top_right']))

    else:
        lower_left = (eval(config['outer_bottom_left']))
        upper_left = (eval(config['outer_top_left']))
        upper_right = (eval(config['outer_top_right']))
        lower_right = (eval(config['outer_bottom_right']))

    boundary_points.append(lower_left)
    boundary_points.append(upper_left)
    boundary_points.append(upper_right)
    boundary_points.append(lower_right)
    return boundary_points


# Function Name: read_config_file
# Description: Read config.ini and store all the parameters in global variables
# Input: (input_video_file: Input video name)
# Output: None. Reads config.ini and store all the parameters in global variables
def read_config_file(input_video_file):

    global roi_points, read_config_file_points, polygon, top_polygon, bottom_polygon
    global frame_rate, distance, speed_threshold

    try:
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read("./data/" + input_video_file + "/config/" + "/config.ini")
        coordinates = config['HSCD']

        frame_rate = float(coordinates['frame_rate'])
        distance = float(coordinates['distance'])

        speed_threshold = int(coordinates['speed_limit'])
        boundary_points = get_roi_points(coordinates, 0)
        boundary_inner_top = get_roi_points(coordinates, 1)
        boundary_inner_bottom = get_roi_points(coordinates, -1)

        polygon = Polygon(boundary_points)
        top_polygon = Polygon(boundary_inner_top)
        bottom_polygon = Polygon(boundary_inner_bottom)

        roi_points = boundary_points
    except KeyError:
        print('\n*** ERROR: Unable to parse config.ini ***')
        print('Try running with \'--set-roi\' argument\n')
        sys.exit(1)

    read_config_file_points = True


# Function Name: get_centroid
# Description: Calculate centroid of a bounding box which consists of (x1,y1,x2,y2)
# Input: (bounding_box: numpy list of x1, y1, x2, y2)
# Output: Returns (x, y) coordinate of centroid
def get_centroid(bounding_box):
    """ Calculate centroid of a bounding box which consists of (x1,y1,x2,y2) """
    # find centroid
    reshape_bbox = np.reshape(bounding_box, (2, 2))
    reshape_bbox[1] = (reshape_bbox[1] - reshape_bbox[0]) / 2
    centroid = np.sum(reshape_bbox, axis=0)
    return centroid


# Function Name: speed_calculate
# Description: Calculate speed, s = d/t
# Input: (total_frames_id: total frames of car in RoI)
# Output: Returns speed of car in kmph
def __speed_calculate(total_frames_id):
    """ Calculate speed, s = d/t """
    total_time = float(total_frames_id / frame_rate)
    speed_in_ms = float(distance / total_time)
    # convert m/s to km/h
    speed_in_kmph = int(round(float(speed_in_ms * 3600 / 1000)))
    return speed_in_kmph


# Function Name: draw_hscd_roi
# Description: Draw roi on a given frame
# Input: (frame: Image on which roi should be drawn)
# Output: None. Draws roi on a given frame
def draw_hscd_roi(frame):
    for position in range(0, 3):
        hscd.draw_line(frame, roi_points[position], roi_points[position+1], (3, 255, 118), 2)
    hscd.draw_line(frame, roi_points[3], roi_points[0], (3, 255, 118), 3)


# Function Name: get_speed
# Description: Estimate speed of car
# Input: (video_file_name: Input video name, frame, frame_id: frame number, id_num: car id, bbox: predicted bounding
#        box points)
# Output: Returns estimated speed of car in kmph
def get_speed(video_file_name, frame, frame_id, id_num, bbox):

    global real_time_tracked_points, output_directory_created

    if not read_config_file_points:
        read_config_file(video_file_name)

    draw_hscd_roi(frame)
    cxy = get_centroid(bbox)

    if polygon.contains(Point(cxy)):
        # print('object id {} and its bb {} inside ROI'.format(id_num, bbox))

        object_id_indices = np.where(real_time_tracked_points[:, 1] == id_num)[0]

        if top_polygon.contains(Point(cxy)):
            if object_id_indices.size == 0:
                if len(real_time_tracked_points) == 1:
                    real_time_tracked_points[0] = frame_id, id_num, cxy[0], cxy[1], 1
                    real_time_tracked_points = np.vstack([real_time_tracked_points, np.zeros((1, 5))])
                    return 0

                real_time_tracked_points = np.vstack([real_time_tracked_points, (frame_id, id_num, cxy[0], cxy[1], 1)])
                real_time_tracked_points = np.vstack([real_time_tracked_points, np.zeros((1, 5))])
                return 0

            if object_id_indices.size == 1:
                point_position = real_time_tracked_points[object_id_indices[0], 4]
                if point_position == -1:
                    real_time_tracked_points[object_id_indices[0] + 1] = frame_id, id_num, cxy[0], cxy[1], 1
                return 0

            if object_id_indices.size == 2:
                real_time_tracked_points[object_id_indices[1]] = frame_id, id_num, cxy[0], cxy[1], 1
                return 0

        elif bottom_polygon.contains(Point(cxy)):
            if object_id_indices.size == 0:
                if len(real_time_tracked_points) == 1:
                    real_time_tracked_points[0] = frame_id, id_num, cxy[0], cxy[1], -1
                    real_time_tracked_points = np.vstack([real_time_tracked_points, np.zeros((1, 5))])
                    return 0

                real_time_tracked_points = np.vstack([real_time_tracked_points, (frame_id, id_num, cxy[0], cxy[1], -1)])
                real_time_tracked_points = np.vstack([real_time_tracked_points, np.zeros((1, 5))])
                return 0

            if object_id_indices.size == 1:
                point_position = real_time_tracked_points[object_id_indices[0], 4]
                if point_position == 1:
                    real_time_tracked_points[object_id_indices[0] + 1] = frame_id, id_num, cxy[0], cxy[1], -1
                return 0

            if object_id_indices.size == 2:
                real_time_tracked_points[object_id_indices[1]] = frame_id, id_num, cxy[0], cxy[1], -1
                return 0

    else:
        object_id_indices = np.where(real_time_tracked_points[:, 1] == id_num)[0]
        if object_id_indices.size == 2:

            total_frames = abs(real_time_tracked_points[object_id_indices[1], 0] -
                            real_time_tracked_points[object_id_indices[0], 0]) + 1
            # print('\n-------------------------------------------------------------------------')
            # print('total frames of id {} is {}'.format(id_num, total_frames))
            speed = __speed_calculate(total_frames)
            # print('speed of id {} is {} kmph\n'.format(id_num, speed))
            cv2.putText(frame, str(id_num), (int(bbox[0]), int(bbox[1]) - 12), 0, 1e-3 * frame.shape[0], (255, 255, 255),
                        int((frame.shape[0] + frame.shape[1]) / 300) / 10)
            cv2.putText(frame, str(speed), (bbox[0] + 50, bbox[1] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        1e-3 * frame.shape[0], (0, 255, 255), int((frame.shape[0] + frame.shape[1]) / 300) / 10)
            return speed
        return 0
