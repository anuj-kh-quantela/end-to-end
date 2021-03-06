import sys
import os

import numpy as np
import cv2
import json
import errno
import matplotlib.pyplot as plt

import skvideo.io

from applications.counting_cars import countingCars
from applications.direction_detection import direction_detection
from applications.speed_estimation import speed_estimation
from applications.traffic_signal_violation_detection import traffic_signal_violation_detection


# from scipy.special import expit
# from utils.box import BoundBox, box_iou, prob_compare
# from utils.box import prob_compare2, box_intersection
from ...utils.box import BoundBox

from ...cython_utils.cy_yolo2_findboxes import box_constructor

direction_detection_result = -1
speed_estimation_result = -1
traffic_signal_violation_result = False

ds = True
try:
	from deep_sort.application_util import preprocessing as prep
	from deep_sort.application_util import visualization
	from deep_sort.deep_sort.detection import Detection
except:
	ds = False



from datetime import datetime

# count_dirs = 0

# def counter():
#     global count_dirs
#     print(count_dirs)
#     count_dirs = count_dirs + 1

def video_writer(v_path, frame_path, img_path, w, h):
	fps = 15
	out = cv2.VideoWriter(v_path+".avi", cv2.VideoWriter_fourcc('M','J','P','G'), fps, (w, h))

	with open(frame_path) as fp:
		for line in fp:
			line = line.strip()
			v_image = cv2.imread(img_path+line)
			out.write(v_image)
	out.release()

	return True

def expit(x):
	return 1. / (1. + np.exp(-x))


def _softmax(x):
	e_x = np.exp(x - np.max(x))
	out = e_x / e_x.sum()
	return out


def findboxes(self, net_out):
	# meta
	meta = self.meta
	boxes = list()
	boxes = box_constructor(meta, net_out)
	return boxes


def extract_boxes(new_im):
	cont = []
	new_im = new_im.astype(np.uint8)
	ret, thresh = cv2.threshold(new_im, 127, 255, 0)
	p, contours, hierarchy = cv2.findContours(
		thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	for i in range(0, len(contours)):
		cnt = contours[i]
		x, y, w, h = cv2.boundingRect(cnt)
		if w * h > 30 ** 2 and (
					(w < new_im.shape[0] and h <= new_im.shape[1]) or (w <= new_im.shape[0] and h < new_im.shape[1])):
			cont.append([x, y, w, h])
	return cont

# active_ids = []
# frames_of_tracks = [[]]

hscd_active_ids_set = set([])
cmwd_active_ids_set = set([])
tsv_active_ids_set = set([])

hscd_processed_ids_set = set([])
cmwd_processed_ids_set = set([])
tsv_processed_ids_set = set([])

def postprocess(self, net_out, im, frame_id=0, csv_file=None, csv=None, mask=None, encoder=None, tracker=None,
				save=False):
	

	# ids_in_current_frame = []
	hscd_ids_in_current_frame_set = set([])
	cmwd_ids_in_current_frame_set = set([])
	tsv_ids_in_current_frame_set = set([])


	# counter()
	video_name = os.path.basename(self.FLAGS.demo)
	video_name = video_name[:-4]

	"""
	Takes net output, draw net_out, save to disk
	"""
	boxes = self.findboxes(net_out)

	# meta
	meta = self.meta
	nms_max_overlap = 0.1
	threshold = meta['thresh']
	colors = meta['colors']
	labels = meta['labels']
	if type(im) is not np.ndarray:
		imgcv = cv2.imread(im)
	else:
		imgcv = im
	h, w, _ = imgcv.shape
	thick = int((h + w) // 300)
	resultsForJSON = []

	if not self.FLAGS.track:
		for b in boxes:
			boxResults = self.process_box(b, h, w, threshold)
			if boxResults is None:
				continue
			left, right, top, bot, mess, max_indx, confidence = boxResults
			if self.FLAGS.json:
				resultsForJSON.append(
					{"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top},
					 "bottomright": {"x": right, "y": bot}})
				continue
			if self.FLAGS.display:
				cv2.rectangle(imgcv,
							  (left, top), (right, bot),
							  colors[max_indx], thick)
				cv2.putText(imgcv, mess, (left, top - 12),
							0, 1e-3 * h, colors[max_indx], thick // 6)
	else:
		if not ds:
			print("ERROR : deep sort or sort submodules not found for tracking please run :")
			print("\tgit submodule update --init --recursive")
			print("ENDING")
			exit(1)
		detections = []
		scores = []

		for b in boxes:
			boxResults = self.process_box(b, h, w, threshold)
			if boxResults is None:
				continue
			left, right, top, bot, mess, max_indx, confidence = boxResults
			if self.FLAGS.trackObj != mess:
				continue
			if self.FLAGS.tracker == "deep_sort":
				detections.append(np.array([left, top, right - left, bot - top]).astype(np.float64))
				scores.append(confidence)
			elif self.FLAGS.tracker == "sort":
				detections.append(np.array([left, top, right, bot]).astype(np.float64))
		if len(detections) < 5 and self.FLAGS.BK_MOG:
			detections = detections + extract_boxes(mask)
		detections = np.array(detections)

		if self.FLAGS.tracker == "deep_sort":
			scores = np.array(scores)
			features = encoder(imgcv, detections.copy())
			detections = [
				Detection(bbox, score, feature) for bbox, score, feature in
				zip(detections, scores, features)]
			# Run non-maxima suppression.
			boxes = np.array([d.tlwh for d in detections])
			scores = np.array([d.confidence for d in detections])
			indices = prep.non_max_suppression(boxes, nms_max_overlap, scores)
			detections = [detections[i] for i in indices]
			tracker.predict()
			tracker.update(detections)
			trackers = tracker.tracks
		elif self.FLAGS.tracker == "sort":
			trackers = tracker.update(detections)

		for track in trackers:
			if self.FLAGS.tracker == "deep_sort":
				if not track.is_confirmed() or track.time_since_update > 1:
					continue
				bbox = track.to_tlbr()
				id_num = str(track.track_id)
			elif self.FLAGS.tracker == "sort":
				bbox = [int(track[0]), int(track[1]), int(track[2]), int(track[3])]
				id_num = str(int(track[4]))

			""" Entry point to all application functions """
			# -------------------------------------------------------------------------
			# -------------------------------------------------------------------------
			""" if counting cars has been chosen by user then this block gets executed PER OBJECT ID AFTER TRACKING """

			if self.FLAGS.counting_cars:
				saveAs = os.path.basename(self.FLAGS.demo)
				saveAs = saveAs[:-4]
				count, line = countingCars.count(int(frame_id), int(id_num), bbox, h)

				# f = open("data/" + saveAs + "/output" + "/car_count_{}.txt".format(saveAs), 'w')
				f = open(self.FLAGS.path_to_output+"/"+self.FLAGS.start_time+"_car_count_{}.txt".format(saveAs), 'w')
				f.write('Cars_Counted: {}'.format(count))

			# ------------------------------------------------------------------------------------------------------
			# ------------------------------------------------------------------------------------------------------

			# video_name = os.path.splitext(os.path.basename(self.FLAGS.demo))[0]

			""" if speed estimation has been chosen by user then this block gets executed PER OBJECT ID AFTER TRACKING """
			if self.FLAGS.speed_estimation:
				global speed_estimation_result
				speed_estimation_result = speed_estimation.get_speed(video_name, np.copy(imgcv), int(frame_id),
																	 int(id_num), [int(bbox[0]), int(bbox[1]),
																				   int(bbox[2]), int(bbox[3])])
				saveAs = os.path.basename(self.FLAGS.demo)
				saveAs = saveAs[:-4]
			# -------------------------------------------------------------------------------------------------------
			# -------------------------------------------------------------------------------------------------------
			""" if direction detection has been chosen by user then this block gets executed PER OBJECT ID AFTER TRACKING """
			if self.FLAGS.direction_detection:
				global direction_detection_result
				direction_detection_result = direction_detection.get_direction(video_name,
																			   np.copy(imgcv), int(frame_id),
																			   int(id_num), [int(bbox[0]), int(bbox[1]),
																							 int(bbox[2]),
																							 int(bbox[3])])
			# ------------------------------------------------------------------------------------------------------
			# ------------------------------------------------------------------------------------------------------
			""" if traffic signal violation detection has been chosen by user then this block gets executed PER OBJECT ID AFTER TRACKING """

			if self.FLAGS.traffic_signal_violation_detection:
				global traffic_signal_violation_result, direction_detection_result
				traffic_signal_violation_result = \
					traffic_signal_violation_detection.detect_red_violation(video_name, imgcv, int(frame_id),
																			int(id_num), [int(bbox[0]), int(bbox[1]),
																						  int(bbox[2]), int(bbox[3])],
																			direction_detection_result)

			# ------------------------------------------------------------------------------------------------------
			# ------------------------------------------------------------------------------------------------------

			if self.FLAGS.csv:
				csv.writerow([frame_id, id_num, int(bbox[0]), int(bbox[1]), int(bbox[2]) - int(bbox[0]),
							  int(bbox[3]) - int(bbox[1])])
				csv_file.flush()

			""" if display flag is enabled, it annotates the car """
			if self.FLAGS.display:

				car_violated = False
				cv2.putText(imgcv, id_num, (int(bbox[0]), int(bbox[1]) - 12), 0, int(1e-3 * h), (255, 255, 255),
							int(thick // 6))

				if self.FLAGS.counting_cars:
					text_size = cv2.getTextSize(str(count), 0, 2, 2)
					pt1 = 90, 180
					pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + text_size[0][1]
					center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
					cv2.line(imgcv, (0, line), (1920, line), (243, 150, 33), 5)
					cv2.rectangle(imgcv, pt1, pt2, (243, 150, 33), -1)
					cv2.putText(imgcv, str(count), center, 0, 2, (34, 87, 255), 2)

				if self.FLAGS.speed_estimation:
					cv2.rectangle(imgcv, (10, 10), (30, 30), (30, 44, 243), -1)
					cv2.putText(imgcv, "speed violation", (40, 30), 0, int(1e-3 * h), (30, 44, 243), int(thick // 4))

				if self.FLAGS.direction_detection:
					cv2.rectangle(imgcv, (10, 40), (30, 60), (249, 0, 213), -1)
					cv2.putText(imgcv, "direction violation", (40, 60), 0, int(1e-3 * h), (249, 0, 213),
								int(thick // 4))

				if self.FLAGS.traffic_signal_violation_detection:
					cv2.rectangle(imgcv, (10, 70), (30, 90), (0, 255, 255), -1)
					cv2.putText(imgcv, "traffic signal violation", (40, 90), 0, int(1e-3 * h), (0, 255, 255),
								int(thick // 4))

				# -------------------------------------------------------------------------------------
				if speed_estimation_result > speed_estimation.speed_threshold:
					cv2.putText(imgcv, str(speed_estimation_result), (int(bbox[0] + 50), int(bbox[1] - 12)),
								cv2.FONT_HERSHEY_SIMPLEX,
								int(1e-3 * h), (0, 255, 255), int(thick // 6))
					cv2.rectangle(imgcv, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
								  (30, 44, 243), thick // 3)
					
					self.__speed_estimation_result = speed_estimation_result
					self.__speed_threshold = speed_estimation.speed_threshold
					
					# 1. extract video name
					saveAs = os.path.basename(self.FLAGS.demo)
					saveAs = saveAs[:-4]

					# 2. make folders object wise
					if not os.path.exists(self.FLAGS.path_to_output + '/hscd/'+self.FLAGS.start_time+'_videos/' + str(id_num)):
						os.makedirs(self.FLAGS.path_to_output + '/hscd/'+self.FLAGS.start_time+'_videos/' + str(id_num))
					
					
					# 3. write frame names to a file 
					vCars = open(str(self.FLAGS.path_to_output) + '/hscd/'+self.FLAGS.start_time+'_videos/' + str(id_num) +"/" + str(self.FLAGS.start_time) +"_{}.txt".format(id_num), 'a')
					vCars.write(str(self.FLAGS.start_time)+'_frame_{}.jpg\n'.format(frame_id))
					vCars.close()

					# 4. write frames
					path = self.FLAGS.path_to_output+'/hscd/'+self.FLAGS.start_time+'_videos/'+str(id_num)
					image_name_string = str(self.FLAGS.start_time) + "_frame_"+ str(frame_id) +".jpg"
					cv2.imwrite(os.path.join(path , image_name_string), imgcv)
					
					# 5. active, history
					hscd_ids_in_current_frame_set.add(id_num)
					hscd_active_ids_set.add(id_num)

					# ids_in_current_frame.append(int(id_num))
					# active_ids.append(int(id_num))
					
					# frames_of_tracks[int(id_num)].append(frame_id)

					data = {
						'module_name': "hscd",
						'location' : self.FLAGS.location_name,
						'tagged_car_id' : int(id_num),
						'time' :  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
						'speed_detected':self.__speed_estimation_result,
						'speed_threshold' : self.__speed_threshold,
						'url' : self.FLAGS.path_to_output + '/hscd/'+self.FLAGS.start_time+'_videos/'+str(id_num)+'/'+str(self.FLAGS.start_time)+"_"+str(id_num)+'.mp4'
					}

					json_file_path = self.FLAGS.path_to_output+'/hscd/'+self.FLAGS.start_time+'_videos/'+str(id_num)
					json_file_name = str(self.FLAGS.start_time)+"_"+str(id_num)+'.json'
					with open(os.path.join(json_file_path, json_file_name), 'w') as f:
						json.dump(data, f)

					car_violated = True



				elif speed_estimation.speed_threshold > speed_estimation_result > 0:
					cv2.putText(imgcv, id_num, (int(bbox[0]), int(bbox[1]) - 12), 0, int(1e-3 * h), (255, 255, 255),
								int(thick // 6))
					cv2.rectangle(imgcv, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
								  (255, 255, 255), thick // 3)
					car_violated = False
				# -------------------------------------------------------------------------------------
				if direction_detection_result == 1:
					cv2.rectangle(imgcv, (int(bbox[0]) + 10, int(bbox[1]) + 10), (int(bbox[2]) - 10, int(bbox[3]) - 10),
								  (249, 0, 213), 4)
					
					# # 1. extract video name
					# saveAs = os.path.basename(self.FLAGS.demo)
					# saveAs = saveAs[:-4]

					# 2. make folders object wise
					if not os.path.exists(self.FLAGS.path_to_output + '/cmwd/'+self.FLAGS.start_time+'_videos/' + str(id_num)):
						os.makedirs(self.FLAGS.path_to_output + '/cmwd/'+self.FLAGS.start_time+'_videos/' + str(id_num))
					
					
					# 3. write frame names to a file 
					vCars = open(str(self.FLAGS.path_to_output) + '/cmwd/'+self.FLAGS.start_time+'_videos/' + str(id_num) +"/" + str(self.FLAGS.start_time) +"_{}.txt".format(id_num), 'a')
					vCars.write(str(self.FLAGS.start_time)+'_frame_{}.jpg\n'.format(frame_id))
					vCars.close()

					# 4. write frames
					path = self.FLAGS.path_to_output+'/cmwd/'+self.FLAGS.start_time+'_videos/'+str(id_num)
					image_name_string = str(self.FLAGS.start_time) + "_frame_"+ str(frame_id) +".jpg"
					cv2.imwrite(os.path.join(path , image_name_string), imgcv)
					
					# 5. active, history
					cmwd_ids_in_current_frame_set.add(id_num)
					cmwd_active_ids_set.add(id_num)

					# ids_in_current_frame.append(int(id_num))
					# active_ids.append(int(id_num))
					
					# frames_of_tracks[int(id_num)].append(frame_id)

					data = {
						'module_name': "cmwd",
						'location' : self.FLAGS.location_name,
						'tagged_car_id' : int(id_num),
						'time' :  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
						'url' : self.FLAGS.path_to_output + '/cmwd/'+self.FLAGS.start_time+'_videos/'+str(id_num)+'/'+str(self.FLAGS.start_time)+"_"+str(id_num)+'.mp4'
					}

					json_file_path = self.FLAGS.path_to_output+'/cmwd/'+self.FLAGS.start_time+'_videos/'+str(id_num)
					json_file_name = str(self.FLAGS.start_time)+"_"+str(id_num)+'.json'
					with open(os.path.join(json_file_path, json_file_name), 'w') as f:
						json.dump(data, f)

					car_violated = True

				# -------------------------------------------------------------------------------------
				if traffic_signal_violation_result:
					cv2.rectangle(imgcv, (int(bbox[0]) - 10, int(bbox[1]) - 10), (int(bbox[2]) + 10, int(bbox[3]) + 10),
								  (0, 255, 255), 4)
					
				
					# 2. make folders object wise
					if not os.path.exists(self.FLAGS.path_to_output + '/tsv/'+self.FLAGS.start_time+'_videos/' + str(id_num)):
						os.makedirs(self.FLAGS.path_to_output + '/tsv/'+self.FLAGS.start_time+'_videos/' + str(id_num))
					
					
					# 3. write frame names to a file 
					vCars = open(str(self.FLAGS.path_to_output) + '/tsv/'+self.FLAGS.start_time+'_videos/' + str(id_num) +"/" + str(self.FLAGS.start_time) +"_{}.txt".format(id_num), 'a')
					vCars.write(str(self.FLAGS.start_time)+'_frame_{}.jpg\n'.format(frame_id))
					vCars.close()

					# 4. write frames
					path = self.FLAGS.path_to_output+'/tsv/'+self.FLAGS.start_time+'_videos/'+str(id_num)
					image_name_string = str(self.FLAGS.start_time) + "_frame_"+ str(frame_id) +".jpg"
					cv2.imwrite(os.path.join(path , image_name_string), imgcv)
					
					# 5. active, history
					tsv_ids_in_current_frame_set.add(id_num)
					tsv_active_ids_set.add(id_num)

					# ids_in_current_frame.append(int(id_num))
					# active_ids.append(int(id_num))
					
					# frames_of_tracks[int(id_num)].append(frame_id)

					data = {
						'module_name': "tsv",
						'location' : self.FLAGS.location_name,
						'tagged_car_id' : int(id_num),
						'time' :  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
						'url' : self.FLAGS.path_to_output + '/tsv/'+self.FLAGS.start_time+'_videos/'+str(id_num)+'/'+str(self.FLAGS.start_time)+"_"+str(id_num)+'.mp4'
					}

					json_file_path = self.FLAGS.path_to_output+'/tsv/'+self.FLAGS.start_time+'_videos/'+str(id_num)
					json_file_name = str(self.FLAGS.start_time)+"_"+str(id_num)+'.json'
					with open(os.path.join(json_file_path, json_file_name), 'w') as f:
						json.dump(data, f)



					car_violated = True

				if not car_violated:
					cv2.rectangle(imgcv, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
								  (255, 255, 255), thick // 3)
	
	
	alertsFile = open(self.FLAGS.path_to_output + "/" +  self.FLAGS.start_time+"_alert_poller.txt", 'a')

	################################################## HSCD #####################################
	# inactive_ids = active_ids - ids_in_current_frame
	hscd_inactive_id_set = hscd_active_ids_set - hscd_ids_in_current_frame_set
	# print(frame_id, hscd_inactive_id_set)
	
	for object_id in hscd_inactive_id_set:

		path = self.FLAGS.path_to_output+'/hscd/'+self.FLAGS.start_time+'_videos/'+str(object_id)+"/"
		# img_path = os.path.join(self.FLAGS.output_dir, path)
		img_path = path
		frame_path = path +str(self.FLAGS.start_time)+ "_{}.txt".format(object_id)
		
		v_path = path + str(self.FLAGS.start_time)+"_" +object_id
		
		if not object_id in hscd_processed_ids_set:
			# print("object_id: {} already present".format(object_id))
			alertsFile.write(self.FLAGS.path_to_output + '/hscd/{}_videos/{}/{}_{}.json\n'.format(self.FLAGS.start_time, object_id, self.FLAGS.start_time, object_id))

			fps = 15
			out = cv2.VideoWriter(v_path+".avi", cv2.VideoWriter_fourcc('M','J','P','G'), fps, (w, h))
			
			with open(frame_path) as fp:
				for line in fp:
					line = line.strip()
					v_image = cv2.imread(img_path+line)
					out.write(v_image)
					if os.path.isfile(img_path+line):
						os.remove(img_path+line)
			out.release()

		else:
			with open(frame_path) as fp:
				for line in fp:
					line = line.strip()
					if os.path.isfile(img_path+line):
						os.remove(img_path+line)


		hscd_processed_ids_set.add(object_id)
		hscd_active_ids_set.remove(object_id)


	# alertsFile.close()

	# inactive_ids = [i for i in active_ids if i not in ids_in_current_frame]

	# inactive_ids = list(set(inactive_ids))
	# print inactive_ids
	
	# for object_id in inactive_ids:

		# print(frames_of_tracks[object_id])
		# draw video corresponding to object_id

		# delete object_id from active_ids
	################################################## HSCD #####################################



	################################################## CMWD #####################################
	cmwd_inactive_id_set = cmwd_active_ids_set - cmwd_ids_in_current_frame_set
	
	for object_id in cmwd_inactive_id_set:
		
		path = self.FLAGS.path_to_output+'/cmwd/'+self.FLAGS.start_time+'_videos/'+str(object_id)+"/"
		img_path = path
		frame_path = path +str(self.FLAGS.start_time)+ "_{}.txt".format(object_id)
		v_path = path + str(self.FLAGS.start_time)+"_" +object_id

		if not object_id in cmwd_processed_ids_set:

			alertsFile.write(self.FLAGS.path_to_output + '/cmwd/{}_videos/{}/{}_{}.json\n'.format(self.FLAGS.start_time, object_id, self.FLAGS.start_time, object_id))
			
			fps = 15
			out = cv2.VideoWriter(v_path+".avi", cv2.VideoWriter_fourcc('M','J','P','G'), fps, (w, h))
			
			with open(frame_path) as fp:
				for line in fp:
					line = line.strip()
					v_image = cv2.imread(img_path+line)
					out.write(v_image)
					if os.path.isfile(img_path+line):
						os.remove(img_path+line)

			out.release()

		else:
			with open(frame_path) as fp:
				for line in fp:
					line = line.strip()
					if os.path.isfile(img_path+line):
						os.remove(img_path+line)

		cmwd_processed_ids_set.add(object_id)
		cmwd_active_ids_set.remove(object_id)

	################################################## CMWD #####################################


	################################################## TSV #####################################
	# inactive_ids = active_ids - ids_in_current_frame
	tsv_inactive_id_set = tsv_active_ids_set - tsv_ids_in_current_frame_set
	# print(frame_id, tsv_inactive_id_set)

	for object_id in tsv_inactive_id_set:
		
		path = self.FLAGS.path_to_output+'/tsv/'+self.FLAGS.start_time+'_videos/'+str(object_id)+"/"
		img_path = path
		frame_path = path +str(self.FLAGS.start_time)+ "_{}.txt".format(object_id)
		v_path = path + str(self.FLAGS.start_time)+"_" +object_id
		
		if not object_id in tsv_processed_ids_set:

			alertsFile.write(self.FLAGS.path_to_output + '/tsv/{}_videos/{}/{}_{}.json\n'.format(self.FLAGS.start_time, object_id, self.FLAGS.start_time, object_id))
			
			fps = 15
			out = cv2.VideoWriter(v_path+".avi", cv2.VideoWriter_fourcc('M','J','P','G'), fps, (w, h))
			
			with open(frame_path) as fp:
				for line in fp:
					line = line.strip()
					v_image = cv2.imread(img_path+line)
					out.write(v_image)
					if os.path.isfile(img_path+line):
						os.remove(img_path+line)

			out.release()

		else:
			with open(frame_path) as fp:
				for line in fp:
					line = line.strip()
					if os.path.isfile(img_path+line):
						os.remove(img_path+line)	

		tsv_processed_ids_set.add(object_id)
		tsv_active_ids_set.remove(object_id)

	

	alertsFile.close()

	################################################## TSV #####################################

	if not save: return imgcv

	outfolder = os.path.join(self.FLAGS.imgdir, 'out')
	img_name = os.path.join(outfolder, os.path.basename(im))
	if self.FLAGS.json:
		textJSON = json.dumps(resultsForJSON)
		textFile = os.path.splitext(img_name)[0] + ".json"
		with open(textFile, 'w') as f:
			f.write(textJSON)
		return

	cv2.imwrite(img_name, imgcv)

	# WHAT I CAME UP WITH 
	
	# # inactive_ids = active_ids - ids_in_current_frame
	# inactive_ids = [i for i in active_ids if i not in ids_in_current_frame]

	# for object_id in inactive_ids:

	# 	# print(frames_of_tracks[object_id])
	# 	print("object_id:", object_id)
	# 	# draw video corresponding to object_id

	# 	# delete object_id from active_ids
