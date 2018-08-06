import cv2,os,sys
os.sys.path.append(os.getcwd()+'/src/')
import argparse
import logging
import time
import ast

import common
import numpy as np
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
import multiprocessing
import time
import cv2
import sys 
import json
import matplotlib.pyplot as plt
import datetime


class AdvancedIntrusionDetection(object):
	
	def __init__(self, video_path, city_name, location_name = " "):
		
		self.__video_channel = video_path
		self.__video_name = os.path.splitext(os.path.basename(video_path))[0]
		
		# clean city name given by user    
		self.__city_name = city_name.strip().lower().replace(" ", "_")
		# clean location name given by user
		self.__location_name = location_name.strip().lower().replace(" ", "_")
		# get absolute path to project directory
		self.__dir_hierarchy = os.getcwd()
		# extract project path and project name from absolute path
		self.__project_path, self.__project_name = os.path.split(self.__dir_hierarchy)

		# list of all the predefined directories to be made 
		predef_dir_structure_list = ['input', 'output', 'log', 'config']
		# define root path for the above directory structure
		predef_dir_structure_path = os.path.join(self.__project_path, self.__city_name, self.__location_name, self.__project_name)

		# make all the directories on the defined root path
		for folder in predef_dir_structure_list:        
			dir_structure = os.path.join(predef_dir_structure_path, folder)

			if not os.path.exists(dir_structure):
				os.makedirs(dir_structure)
		

		self.__path_to_output = os.path.join(predef_dir_structure_path, 'output')
		self.__path_to_log = os.path.join(predef_dir_structure_path, 'log')
		

		# create output directory if doesn't exists already
		if not os.path.exists(self.__path_to_output):
			os.makedirs(self.__path_to_output)

		# injecting schema here
		# sensor meta info
		json_file_path = predef_dir_structure_path

		SensorMetaInfo = {
			'CameraID' : self.__video_name,
			'ServerId' : 'vijaywada_PC_01',
			'Product_category_id' : 1,
			'Feature_id' : 1,
			'Lx' : '10.233N',
			'Ly' :  '70.1212S'

		}

		sensor_meta_info_json_file_name = 'SensorMetaInfo.json'
		with open(os.path.join(json_file_path, sensor_meta_info_json_file_name), 'w') as f:
			json.dump(SensorMetaInfo, f)

		# event info
		Event = {
		'Alert_id' : 1,
		'TypeDescription' : 'Somebody enter the virtual fencing'
		}

		event_json_file_name = 'event.json'
		with open(os.path.join(json_file_path, event_json_file_name), 'w') as f:
			json.dump(Event, f)


		print("Done Init!")

	
	def check_intersection(self, a, b):
		"""
		a : rectangle in (x,y,w,h)
		"""
		x = max(a[0], b[0])
		y = max(a[1], b[1])
		w = min(a[0]+a[2], b[0]+b[2]) - x
		h = min(a[1]+a[3], b[1]+b[3]) - y
		if w<0 or h<0: 
			return False,() # or (0,0,0,0) ?
		return True,(x, y, w, h)




	def detect_intrusion(self, input_shape, user_bbox=None, channel=2, zoom=1, model_name='mobilenet_thin', increase_fps=None):
		
		#-------------- fps increment settings ---
		inc_fps_flag, num_frames_to_skip = (increase_fps)
			
		if inc_fps_flag:
			if num_frames_to_skip < 1:
				print('entered frames to skip is less than 1, please input a higher value')
				sys.exit()
			if num_frames_to_skip > 7:
				print('entered frames to skip is more than 7, please input a value less than 7')
				sys.exit()
		#-------------- fps increment settings ---
		
		
		
		#--------------- intrusion parameters ----------
		text = 'select minimum area object \n skip frame : press s\nconfirmation : press c'
		#--------------- intrusion parameters ----------
		
		
		# pose estimator parameters
		# channel = channel
		zoom = zoom
		im_w, im_h = input_shape
		e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(im_w, im_h))
		
		
		bbox = user_bbox

		out = None
		fourcc = cv2.VideoWriter_fourcc(*'XVID')

		intrusion_started_time = None
		countdown_time = 0
		roi_created_flag = False
		
		intrusion_check = None
		event_identifier = 1

		cap = cv2.VideoCapture(self.__video_channel)

		while cap.isOpened():
			
			###########################################
			# very important setting for writing output 

			current_date = str(datetime.datetime.now().date())
			dated_output_path = os.path.join(self.__path_to_output, current_date)

			dated_video_path = os.path.join(dated_output_path, 'videos')
			dated_alert_path = os.path.join(dated_output_path, 'alerts')

			# create output directory if doesn't exists already
			if not os.path.exists(dated_video_path):
				os.makedirs(dated_video_path)

			if not os.path.exists(dated_alert_path):
				os.makedirs(dated_alert_path)


			#############################################

			if inc_fps_flag:
				for i in range(num_frames_to_skip):
					cap.read()
			

			try:
				ret, image = cap.read()
				
				####################### start  ########################

				#----------------------------------- ROI selector ---------------------------
				while bbox is None:
					cv2.namedWindow('select ROI', cv2.WINDOW_NORMAL)
					for idx,i in enumerate(text.split('\n')):
						cv2.putText(image, i, (10,40*(idx+1)), cv2.FONT_HERSHEY_PLAIN, 1,(0,255,0),1,cv2.LINE_AA)


					k = cv2.waitKey(1)
					cv2.imshow('select ROI', image)
					if k==115:
						##  press s
						bbox = None
						ret, image = cap.read()
						
					if k==99:
						## press c
						bbox = cv2.selectROI('select ROI', image)
						cv2.destroyWindow('select ROI')
						if not any(bbox):
							print("NO ROI SELECTED, Exiting!")
							sys.exit(0)
						print("Selected ROI Coordinates: " + str(bbox))
				#----------------------------------- ROI selector ---------------------------
				
					
				if zoom < 1.0:
					canvas = np.zeros_like(image)
					img_scaled = cv2.resize(image, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
					dx = (canvas.shape[1] - img_scaled.shape[1]) // 2
					dy = (canvas.shape[0] - img_scaled.shape[0]) // 2
					canvas[dy:dy + img_scaled.shape[0], dx:dx + img_scaled.shape[1]] = img_scaled
					image = canvas
				elif zoom > 1.0:
					img_scaled = cv2.resize(image, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
					dx = (img_scaled.shape[1] - image.shape[1]) // 2
					dy = (img_scaled.shape[0] - image.shape[0]) // 2
					image = img_scaled[dy:image.shape[0], dx:image.shape[1]]


				# detect human using tf-pose
				humans = e.inference(image)
				

				# modified tf-pose-code
				# get all points of a human 
				image, centers_humans = e.draw_humans(image, humans, imgcopy=False)
				
				rect_array = []
				# find the bounding rectangle out of the human points
				for centers in centers_humans.values():
					rect = cv2.boundingRect(np.array( centers.values()))
					rect_array.append(rect)
					(x,y,w,h) = rect
					image = cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),3)
				
				
				res = []
				for rect in rect_array :
					res.append(self.check_intersection(np.array(rect), np.array(bbox))[0])
				
				cv2.rectangle(image, (bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(255,0,0),3)

				if any(res):
					if ((intrusion_started_time is None) and (countdown_time==0)):
						
						intrusion_started_time = datetime.datetime.now()
						
						video_name = "intrusion_"+str(datetime.datetime.now().time().strftime('%H:%M:%S')) + ".avi"
						abs_video_path = os.path.join(dated_video_path, video_name)
						out = cv2.VideoWriter(abs_video_path, fourcc, 15.0, (image.shape[1], image.shape[0]))

						# video_name = os.path.join(dated_video_path, intrusion_started_time.strftime("%Y_%m_%d_%H_%M_%S")+'.avi')
						# out = cv2.VideoWriter(video_file_name, fourcc, 15.0, (image.shape[1],image.shape[0]))
						
						intrusion_check = True
					countdown_time = 15  # extra time fow which video is going to be written
					
					out.write(image)

					num_humans = sum(res)
					cv2.putText(image,"Number of humans: "+str(num_humans),(40,40), cv2.FONT_HERSHEY_PLAIN, 2,(0, 0, 0),2, cv2.LINE_AA)
					
					
					human_list = []
					for idx, (xA, yA, xB, yB) in enumerate(rect_array):
						
						human_list.append({
								"name" : "human_"+str(idx),
								"Point1": { "X": str(xA), "Y": str(yA) }, 
								"Point2": { "X": str(xA+xB), "Y": str(yA) }, 
								"Point3": { "X": str(xA), "Y": str(yA+yB) },  
								"Point4": { "X": str(xB), "Y": str(yB) }

							})


					cdata = { "input": { "data": { "SensorMetaInfo": {"CameraID": "test_video_1", "Feature_id": 1, "Product_category_id": 1, "ServerId": "vijaywada_PC_01", "Lx": "10.233N", "Ly": "70.1212S", "CameraDescription": "cisco_cam_type_1", "LongDescription": "low range camera"}, "Event": { "EventIdentifier": event_identifier, "EventID": 1, "EventDescription": "Somebody enter the virtual fencing" }, "ROI_drawn": { "Point1": { "X": str(bbox[0]), "Y": str(bbox[1]) }, "Point2": { "X": str(bbox[0]+bbox[2]), "Y": str(bbox[1]) }, "Point3": { "X": str(bbox[0]), "Y": str(bbox[1]+bbox[3]) }, "Point4": { "X": str(bbox[0]+bbox[2]), "Y": str(bbox[1]+bbox[3]) } }, "Data": { "number_of_humans": str(num_humans), 

					"detected_roi": human_list,

						"CapturedTime": str(datetime.datetime.now()), "VideoURL": abs_video_path } } }, "configName": "IntrusionDetection", "groupName": "VideoAnalytics" }



					if intrusion_check:
						# print("\n==== check for intrusion -----")
						# print(cdata)
						# dump json 
						json_file_name = "intrusion_alert_"+str(datetime.datetime.now().time().strftime('%H:%M:%S')) + ".json"
						abs_json_file_path = os.path.join(dated_alert_path, json_file_name)
						with open(abs_json_file_path, 'w') as f:
							json.dump(cdata, f)
						pass
						# ########################## UNCOMMENT THIS LINE FOR IN ORDER TO PUSH TO API END POINT ################
						# subprocess.call(["curl", "-X", "POST", "http://52.74.189.153:9090/api/v1/source/getInputData", "-H", "Cache-Control: no-cache", "-H", "Content-Type: application/json", "-H", "Postman-Token: 8a74ff29-c6cd-48ef-ad48-78a85c66ff94", "-H", "x-access-token: MW7VN68RJAFJ0K5XPRZPKOPN02RDK9JR", "-d", json.dumps(cdata)])
						# ########################## UNCOMMENT THIS LINE FOR IN ORDER TO PUSH TO API END POINT ################

						# print("\n")

					intrusion_check = False


				else :
					intrustion_stopped = True
					intrusion_started_time = None
					if out is not None:
						countdown_time-=1
						# print("countdown_time", countdown_time)
						if (countdown_time==0 ):
							# print('COUNT DOWN ERROR')
							# print(json.dumps(cdata))
							# subprocess.call(['curl -X POST http://52.74.189.153:9090/api/v1/source/getInputData', '-H', 'Cache-Control: no-cache', '-H', 'Content-Type: application/json',  ], shell=True)
							# subprocess.call(['curl -X POST   http://52.74.189.153:9090/api/v1/source/getInputData', '-H', 'Cache-Control: no-cache', '-H', 'Content-Type: application/json', '-H', 'Postman-Token: 8a74ff29-c6cd-48ef-ad48-78a85c66ff94', '-H', 'x-access-token: MW7VN68RJAFJ0K5XPRZPKOPN02RDK9JR', '-d', '{ "input": { "data": { "SensorMetaInfo": {"CameraID": "test_video_1", "Feature_id": 1, "Product_category_id": 1, "ServerId": "vijaywada_PC_01", "Lx": "10.233N", "Ly": "70.1212S", "CameraDescription": "cisco_cam_type_1", "LongDescription": "low range camera"}, "Event": { "EventID": 1, "EventDescription": "Somebody enter the virtual fencing" }, "ROI_drawn": { "Point1": { "X": "10", "Y": "132" }, "Point2": { "X": "26", "Y": "132" }, "Point3": { "X": "26", "Y": "148" }, "Point4": { "X": "10", "Y": "148" } }, "Data": { "number_of_humans": 2, "detected_roi": { "Point1": { "X": "112", "Y": "312" }, "Point2": { "X": "34", "Y": "356" } }, "CapturedTime": "2018-02-26T10:23:51", "VideoURL": "http://<ip-address>/<path-to-output>/bangalore/indranagar/society-2/intrusion/output visual_files/2018_04_23_18_01/2018_04_23_18_01_11.avi" } } }, "configName": "IntrusionDetection", "groupName": "VideoAnalytics" }'], shell=True)
							# subprocess.call(["curl", "-X", "POST", "http://52.74.189.153:9090/api/v1/source/getInputData", "-H", "Cache-Control: no-cache", "-H", "Content-Type: application/json", "-H", "Postman-Token: 8a74ff29-c6cd-48ef-ad48-78a85c66ff94", "-H", "x-access-token: MW7VN68RJAFJ0K5XPRZPKOPN02RDK9JR", "-d", '{ "input": { "data": { "SensorMetaInfo": {"CameraID": "test_video_1", "Feature_id": 1, "Product_category_id": 1, "ServerId": "vijaywada_PC_01", "Lx": "10.233N", "Ly": "70.1212S", "CameraDescription": "cisco_cam_type_1", "LongDescription": "low range camera"}, "Event": { "EventID": 1, "EventDescription": "Somebody enter the virtual fencing" }, "ROI_drawn": { "Point1": { "X": "10", "Y": "132" }, "Point2": { "X": "26", "Y": "132" }, "Point3": { "X": "26", "Y": "148" }, "Point4": { "X": "10", "Y": "148" } }, "Data": { "number_of_humans": 2, "detected_roi": { "Point1": { "X": "112", "Y": "312" }, "Point2": { "X": "34", "Y": "356" } }, "CapturedTime": "2018-02-26T10:23:51", "VideoURL": "http://<ip-address>/<path-to-output>/bangalore/indranagar/society-2/intrusion/output visual_files/2018_04_23_18_01/2018_04_23_18_01_11.avi" } } }, "configName": "IntrusionDetection", "groupName": "VideoAnalytics" }'])
							
							# ########################## UNCOMMENT THIS LINE FOR IN ORDER TO PUSH TO API END POINT ################
							# subprocess.call(["curl", "-X", "POST", "http://52.74.189.153:9090/api/v1/source/getInputData", "-H", "Cache-Control: no-cache", "-H", "Content-Type: application/json", "-H", "Postman-Token: 8a74ff29-c6cd-48ef-ad48-78a85c66ff94", "-H", "x-access-token: MW7VN68RJAFJ0K5XPRZPKOPN02RDK9JR", "-d", json.dumps(cdata)])
							# ########################## UNCOMMENT THIS LINE FOR IN ORDER TO PUSH TO API END POINT ################

							event_identifier += 1
							
							json_file_name = "intrusion_alert_"+str(datetime.datetime.now().time().strftime('%H:%M:%S')) + ".json"
							abs_json_file_path = os.path.join(dated_alert_path, json_file_name)
							with open(abs_json_file_path, 'w') as f:
								json.dump(cdata, f)
							
							# print("\n")
							out.release()	
							out = None

				
				####################### end  ########################
				cv2.imshow('intrusion', image)
				k = cv2.waitKey(1)
				if k == 27:
					break

			except Exception as e:
				print(e)
				break

		if out is not None:
			out.release()
		cap.release()
		cv2.destroyAllWindows()



video_path = '/media/anuj/Work-HDD/WORK/CLOUD-DRIVE/Google-Drive/Computer-Vision/Sample-Videos/Slip-And-Fall/test-4.webm'
user_bbox = (466, 109, 421, 611)
obj = AdvancedIntrusionDetection(video_path, city_name='vijaywada', location_name='test-loc')
obj.detect_intrusion(input_shape=(640, 352), increase_fps=(True, 6), user_bbox=user_bbox)