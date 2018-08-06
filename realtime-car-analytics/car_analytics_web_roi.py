import os
import sys

import configparser
import errno

from darkflow.darkflow.defaults import argHandler  # Import the default arguments
from darkflow.darkflow.net.build import TFNet

# from applications.roi.direction_detection import draw_custom_roi_wd
# from applications.roi.speed_estimation import draw_custom_roi_hscd
# from applications.roi.traffic_signal_violation_detection import draw_custom_roi_tsvd

from apps.roi.speed_estimation import draw_custom_roi_hscd
from apps.roi.direction_detection import draw_custom_roi_wd
from apps.roi.traffic_signal_violation_detection import draw_custom_roi_tsvd


import tensorflow as tf
import glob
import shutil
from datetime import datetime
import cv2
import numpy as np
import shutil
import json

def clean_names(mod_string):
	return mod_string.strip().lower().replace(" ", "_")

# Class Name: CarAnalytics
# Description: Class containing all flags to enable car analytics tasks
class CarAnalytics:
	# Function Name: __init__
	# Description: enables flags based on user input for car analytics tasks
	# Input: (video_path: path to input video)
	# def __init__(self, video_path, city_name, actual_name = " "):
	def __init__(self, video_path, city_name, location_name = " "):
		
		self.__city_name = city_name.strip().lower().replace(" ", "_")
		self.__location_name = location_name.strip().lower().replace(" ", "_")
		
		self.__dir_hierarchy = os.getcwd()
		# dir_hierarchy = self.__dir_hierarchy.split('/')
		
		self.__project_path, self.__project_name = os.path.split(self.__dir_hierarchy)
		print self.__project_path
		print self.__project_name
		predef_dir_structure_path = os.path.join(self.__project_path, self.__city_name, self.__location_name, self.__project_name)
		
		

		predef_dir_structure_list = ['input', 'output', 'log', 'config']
		for folder in predef_dir_structure_list:		
			dir_structure = os.path.join(predef_dir_structure_path, folder)

			if not os.path.exists(dir_structure):
				os.makedirs(dir_structure)

		
		print "OUTPUT PATH: " +str(os.path.join(predef_dir_structure_path, 'output'))
		print "CONFIG PATH: " +str(os.path.join(predef_dir_structure_path, 'config'))
		print "LOG PATH: " +str(os.path.join(predef_dir_structure_path, 'log'))
		# 2. path to config
		# path_to_config = '../../../config/config.ini'
		sys.exit(0)

		if not os.path.exists(path_to_config):
			config = configparser.ConfigParser()
			config.optionxform = str
			config[self.__project_name +"_"+self.__city_name] = {}
			config[self.__project_name +"_"+self.__city_name]['user_input_video_name'] = " "
			config[self.__project_name +"_"+self.__city_name]['user_output_path'] = " "
			config[self.__project_name +"_"+self.__city_name]['distance'] = "10"
			config[self.__project_name +"_"+self.__city_name]['frame_rate'] = "15"
			config[self.__project_name +"_"+self.__city_name]['speed_limit'] = "10"
			config[self.__project_name +"_"+self.__city_name]['roi_image_path'] = "/tmp/"


			with open(path_to_config, 'w') as configfile:
				config.write(configfile)

		else: 
			config = configparser.ConfigParser()
			config.optionxform = str
			config.read(path_to_config)

			if not config.has_section(self.__project_name +"_"+self.__city_name):
				config[self.__project_name +"_"+self.__city_name] = {}
				config[self.__project_name +"_"+self.__city_name]['user_input_video_name'] = " "
				config[self.__project_name +"_"+self.__city_name]['user_output_path'] = " "	
				config[self.__project_name +"_"+self.__city_name]['distance'] = "10"
				config[self.__project_name +"_"+self.__city_name]['frame_rate'] = "15"
				config[self.__project_name +"_"+self.__city_name]['speed_limit'] = "10"	
				config[self.__project_name +"_"+self.__city_name]['roi_image_path'] = "/tmp/"


				with open(path_to_config, 'w') as configfile:
					config.write(configfile)


		config = configparser.ConfigParser()
		config.optionxform = str
		config.read(path_to_config)
		config_params = config[self.__project_name +"_"+self.__city_name]
		self.__user_input_video_name = str(config_params['user_input_video_name'])
		self.__user_output_path = str(config_params['user_output_path'])
		
		self.__distance = int(config_params['distance'])
		self.__frame_rate = int(config_params['frame_rate'])
		self.__speed_limit = int(config_params['speed_limit'])

		self.__roi_image_path = config_params['roi_image_path']

		# print self.__distance
		# print self.__frame_rate
		# print self.__speed_limit


		self.__video_name = os.path.splitext(os.path.basename(video_path))[0]
		self.__video_path = video_path



		# 3.
		# video_info = os.path.splitext(os.path.basename(video_path))
		# video_name = video_info[0]
		# video_extension = video_info[1]
		
		# self.__video_path = video_path

		# if not os.path.exists(self.video_path):
		#     print("\nVideo file does not exist. Please make sure path and video filename is proper.\n\n***ABORTING***\n")
		#     sys.exit(0)
		
		# actual_name = actual_name.strip()
		# if not (actual_name == ""):
		# 	self.__video_name = actual_name
		# else:
		# 	self.__video_name = os.path.splitext(os.path.basename(video_path))[0]

		
		# if not (self.__user_input_video_name == ""):
		# 	self.__path_to_output = self.__user_output_path + "output/" + self.__city_name +"/"+ self.__project_name + "/"+ self.__user_input_video_name 
		# else:
		# 	self.__path_to_output = self.__user_output_path + "output/" + self.__city_name +"/"+ self.__project_name + "/"+ self.__video_name
		
		# set output path
		self.__user_output_path = self.__user_output_path.strip()
		
		if not self.__user_output_path == "":
			self.__path_to_output = self.__user_output_path + "output/" + self.__city_name +"/"+ self.__project_name + "/"+ self.__video_name
		else:
			self.__user_output_path = os.getcwd()
			self.__path_to_output = self.__user_output_path + "/output/" + self.__city_name +"/"+ self.__project_name + "/"+ self.__video_name


		# set location name
		self.__user_input_video_name = self.__user_input_video_name.strip()
		if not self.__user_input_video_name == "":
			self.__location_name = self.__user_input_video_name
		else:
			self.__location_name = self.__video_name
		
		# HERE IF SOMEONE SPECIFICALLY GIVES THE LOCATION NAME VIA API THEN OVERWRITE THE ONE OBTAINED FROM 
		# CONFIG FILE

		# actual_name = actual_name.strip()
		# if not actual_name == "":
		# 	self.__location_name = actual_name
		# else:
		# 	self.__location_name = self.__video_name



		# self.__path_to_output = self.__user_output_path + "output/" 
		
		# create output directory if doesn't exists already
		if not os.path.exists(self.__path_to_output):
			os.makedirs(self.__path_to_output)

		# if not os.path.exists(self.__path_to_output+self.__video_name):
			# os.makedirs(self.__path_to_output+self.__video_name)

		self.__run_counting_cars = False
		self.__run_detect_direction = False
		self.__run_hscd = False
		self.__run_traffic_violation_detection = False
		self.__start_time = str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
		print('\n***Initialization Complete!***\n')
		print('\nPlease proceed for individual module setup\n')

		
		# Create a folder named video file in data folder
		self.__dir_name = "./data/" + self.__video_name + "/config/"
		# if not os.path.exists(self.__dir_name):
		#     try:
		#         os.makedirs(self.__dir_name)
		#     except OSError as e:
		#         if e.errno == errno.EEXIST:
		#             print("Warning: Directory already exists.\n")
		#         else:
		#             print("Can't create destination directory (%s)!" % self.__dir_name)
		#             sys.exit(0)

		if not os.path.exists(self.__dir_name):
			os.makedirs(self.__dir_name)

		config = configparser.ConfigParser()
		config.optionxform = str
		# # this doesn't gives any error even if the file doesn't exists! strange
		config.read(self.__dir_name + 'config.ini')


	# this remains the same in the NEW as well as the OLD API
	def setup_counting_cars(self):
		self.__run_counting_cars = True


	# # PART OF OLD API 
	# def setup_high_speed_detection(self, pDistance="", pFrameRate="", pSpeedLimit=""):
	# 	if pDistance == "":
	# 		pDistance = self.__distance

	# 	if pFrameRate == "":
	# 		pFrameRate = self.__frame_rate

	# 	if pSpeedLimit == "":
	# 		pSpeedLimit = self.__speed_limit

	# 	draw_custom_roi_hscd.define_roi(self.__video_path, pDistance, pFrameRate, pSpeedLimit)
		
	# 	self.__run_hscd = True

		
	# 	# if not os.path.exists('data/' + self.__video_name + "/output/hscd/videos/"):
	# 		# os.makedirs('data/' + self.__video_name + "/output/hscd/videos/")

	# 	if not os.path.exists(self.__path_to_output + '/hscd/'+ self.__start_time +'_videos/'):
	# 		os.makedirs(self.__path_to_output + '/hscd/'+ self.__start_time +'_videos/')

	# # PART OF NEW API 
	# def new_setup_high_speed_detection(self, pDistance="", pFrameRate="", pSpeedLimit=""):
	# 	if pDistance == "":
	# 		pDistance = self.__distance

	# 	if pFrameRate == "":
	# 		pFrameRate = self.__frame_rate

	# 	if pSpeedLimit == "":
	# 		pSpeedLimit = self.__speed_limit

	# 	user_boundary_points = [(249, 505), (587, 275), (1417, 276), (1746, 515)]

	# 	draw_custom_roi_hscd.define_roi(self.__video_path, pDistance, pFrameRate, pSpeedLimit, user_boundary_points)
		
	# 	self.__run_hscd = True


	# 	if not os.path.exists(self.__path_to_output + '/hscd/'+ self.__start_time +'_videos/'):
	# 		os.makedirs(self.__path_to_output + '/hscd/'+ self.__start_time +'_videos/')

	def hscd_roi_setup(self):
		path_to_hscd_instruction_img = self.__dir_hierarchy + "/" +"RoI_Instruction_Image_HSCD.jpg"
		hscd_roi_image_name = "hscd_roi_image.jpg"
		hscd_roi_image = self.__roi_image_path + hscd_roi_image_name

		cap = cv2.VideoCapture(self.__video_path)
		ret, frame = cap.read()
		cv2.imwrite(hscd_roi_image, frame)
		cap.release()

		data = {
				# if UI requires server path then the caller should append the server path
				'path_to_hscd_instruction_img': path_to_hscd_instruction_img,
				'path_to_hscd_roi_image' : hscd_roi_image
			}

		return json.dumps(data)

	def get_hscd_roi_parameters(self, pDistance="", pFrameRate="", pSpeedLimit="", hscd_json_coords_from_UI=""):
		# if hscd_json_coords_from_UI == "":
		# 	print("ROI setup incomplete, please call hscd_roi_setup() and then run this function. Exiting....")
		# 	sys.exit(0)

		# ###############  this is temporary thing, hscd_json_coords_from_UI is to be read here
		hscd_json_coords_from_UI = {
			"path_to_hscd_roi_image": "/tmp/hscd_roi_image.jpg",
			"user_boundary_points": [
				[249, 505],
				[587, 275],
				[1417, 276],
				[1746, 515]
				]
			}

		temp = json.dumps(hscd_json_coords_from_UI)

		# ###############  Till here this is temporary thing, hscd_json_coords_from_UI is to be read here
		
		

		load_boundary_points = json.loads(temp)


		# load_boundary_points = json.loads(hscd_json_coords_from_UI)
		user_boundary_points = [tuple(pts) for pts in load_boundary_points['user_boundary_points']]

		if pDistance == "":
			pDistance = self.__distance

		if pFrameRate == "":
			pFrameRate = self.__frame_rate

		if pSpeedLimit == "":
			pSpeedLimit = self.__speed_limit

		draw_custom_roi_hscd.define_roi(self.__video_path, pDistance, pFrameRate, pSpeedLimit, user_boundary_points)
		
		self.__run_hscd = True


		if not os.path.exists(self.__path_to_output + '/hscd/'+ self.__start_time +'_videos/'):
			os.makedirs(self.__path_to_output + '/hscd/'+ self.__start_time +'_videos/')

		# once everything is done, trigger response to the UI
		response = ""
		response_to_UI = { "response_from_API": response }

		if self.__run_hscd:
			response = "true"
		else: 
			response = "false"

		response_to_UI = {"response_from_API" : response}
		return json.dumps(response_to_UI)


	# def setup_wrong_direction_detection(self):
	# 	draw_custom_roi_wd.define_roi(self.__video_path)
	# 	self.__run_detect_direction = True

		
	# 	# if not os.path.exists('data/' + self.__video_name + "/output/cmwd/videos/"):
	# 		# os.makedirs('data/' + self.__video_name + "/output/cmwd/videos/")

	# 	if not os.path.exists(self.__path_to_output + '/cmwd/'+ self.__start_time +'_videos/'):
	# 		os.makedirs(self.__path_to_output + '/cmwd/'+ self.__start_time +'_videos/')


	def cmwd_roi_setup(self):
		path_to_cmwd_instruction_img = self.__dir_hierarchy + "/" +"RoI_Instruction_Image_WD.jpg"
		cmwd_roi_image_name = "cmwd_roi_image.jpg"
		cmwd_roi_image = self.__roi_image_path + cmwd_roi_image_name

		cap = cv2.VideoCapture(self.__video_path)
		ret, frame = cap.read()
		cv2.imwrite(cmwd_roi_image, frame)
		cap.release()

		data = {
				# if UI requires server path then the caller should append the server path
				'path_to_cmwd_instruction_img': path_to_cmwd_instruction_img,
				'path_to_cmwd_roi_image' : cmwd_roi_image
			}

		return json.dumps(data)


	def get_cmwd_roi_parameters(self, cmwd_json_coords_from_UI=""):
		# if cmwd_json_coords_from_UI == "":
		# 	print("ROI setup incomplete, please call cmwd_roi_setup() and then run this function. Exiting....")
		# 	sys.exit(0)

		# ###############  this is temporary thing, cmwd_json_coords_from_UI is to be read here
		cmwd_roi_coordinates = {
			"path_to_cmwd_roi_image": "/tmp/cmwd_roi_image.jpg",
			"user_boundary_points": [
				[245, 508],
				[591, 276],
				[1414, 273],
				[1738, 510]
				]
			}

		temp = json.dumps(cmwd_roi_coordinates)

		# ###############  Till here this is temporary thing, cmwd_json_coords_from_UI is to be read here

		load_boundary_points = json.loads(temp)

		# load_boundary_points = json.loads(cmwd_json_coords_from_UI)
		user_boundary_points = [tuple(pts) for pts in load_boundary_points['user_boundary_points']]
		
		draw_custom_roi_wd.define_roi(self.__video_path, user_boundary_points)
		
		self.__run_detect_direction = True

		if not os.path.exists(self.__path_to_output + '/cmwd/'+ self.__start_time +'_videos/'):
			os.makedirs(self.__path_to_output + '/cmwd/'+ self.__start_time +'_videos/')


		# once everything is done, trigger response to the UI
		response = ""
		response_to_UI = { "response_from_API": response }

		if self.__run_detect_direction:
			response = "true"
		else: 
			response = "false"

		response_to_UI = {"response_from_API" : response}
		return json.dumps(response_to_UI)


	def tsv_roi_setup(self):
		path_to_ts_instruction_img = self.__dir_hierarchy + "/" +"RoI_Instruction_Image_RL.jpg"
		path_to_tsv_instruction_img = self.__dir_hierarchy + "/" +"RoI_Instruction_Image_VZ.jpg"
		
		ts_roi_image_name = "ts_roi_image.jpg"
		tsv_roi_image_name = "tsv_roi_image.jpg"
		ts_roi_image = self.__roi_image_path + ts_roi_image_name
		tsv_roi_image = self.__roi_image_path + tsv_roi_image_name

		cap = cv2.VideoCapture(self.__video_path)
		ret, frame = cap.read()
		cv2.imwrite(ts_roi_image, frame)
		cv2.imwrite(tsv_roi_image, frame)
		cap.release()

		# need to tell the UI people that there are 2 different popups that are to be taken care of
		# they should ensure that the user compulsarily needs to select 2 roi(s)

		data = {
				# if UI requires server path then the caller should append the server path
				'path_to_ts_instruction_img': path_to_ts_instruction_img,
				'path_to_tsv_instruction_img': path_to_tsv_instruction_img,
				'path_to_ts_roi_image' : ts_roi_image,
				'path_to_tsv_roi_image' : tsv_roi_image
			}

		return json.dumps(data)


	def get_tsv_roi_parameters(self, ts_json_coords_from_UI="", tsv_json_coords_from_UI=""):
		# if ts_json_coords_from_UI == "" and tsv_json_coords_from_UI == "":
		# 	print("ROI setup incomplete, please call cmwd_roi_setup() and then run this function. Exiting....")
		# 	sys.exit(0)

		# ###############  this is temporary thing, ts_json_coords_from_UI is to be read here
		ts_roi_coordinates = {
			"path_to_ts_roi_image": "/tmp/ts_roi_image.jpg",
			"user_boundary_points": [
				[415, 161],
				[412, 63],
				[455, 64],
				[456, 163]
				]
			}

		ts_temp = json.dumps(ts_roi_coordinates)

		tsv_roi_coordinates = {
			"path_to_tsv_roi_image": "/tmp/tsv_roi_image.jpg",
			"user_boundary_points": [
				[192, 519],
				[481, 335],
				[743, 367],
				[782, 613]
				]
			}

		tsv_temp = json.dumps(tsv_roi_coordinates)

		# ###############  Till here this is ts_temporary thing, ts_json_coords_from_UI is to be read here

		ts_load_boundary_points = json.loads(ts_temp)
		tsv_load_boundary_points = json.loads(tsv_temp)

		# load_boundary_points = json.loads(cmwd_json_coords_from_UI)
		ts_user_boundary_points = [tuple(pts) for pts in ts_load_boundary_points['user_boundary_points']]
		tsv_user_boundary_points = [tuple(pts) for pts in tsv_load_boundary_points['user_boundary_points']]
		
		
		draw_custom_roi_tsvd.define_roi(self.__video_path, 'light', ts_user_boundary_points)
		draw_custom_roi_tsvd.define_roi(self.__video_path, 'zone', tsv_user_boundary_points)
		self.__run_traffic_violation_detection = True
		self.__run_detect_direction = True

		if not os.path.exists(self.__path_to_output + '/tsv/'+ self.__start_time +'_videos/'):
			os.makedirs(self.__path_to_output + '/tsv/'+ self.__start_time +'_videos/')

		# once everything is done, trigger response to the UI
		response = ""
		response_to_UI = { "response_from_API": response }

		if self.__run_traffic_violation_detection and self.__run_detect_direction:
			response = "true"
		else: 
			response = "false"

		response_to_UI = {"response_from_API" : response}
		return json.dumps(response_to_UI)

		
	def setup_traffic_signal_violation_detection(self):
		draw_custom_roi_tsvd.define_roi(self.__video_path, 'light')
		draw_custom_roi_tsvd.define_roi(self.__video_path, 'zone')
		self.__run_traffic_violation_detection = True
		self.__run_detect_direction = True

		# if not os.path.exists('data/' + self.__video_name + "/output/tsv/videos/"):
		# 	os.makedirs('data/' + self.__video_name + "/output/tsv/videos/")

		if not os.path.exists(self.__path_to_output + '/tsv/'+ self.__start_time +'_videos/'):
			os.makedirs(self.__path_to_output + '/tsv/'+ self.__start_time +'_videos/')

	def show_output_path(self):
		if not (self.__run_counting_cars or self.__run_detect_direction or self.__run_hscd or self.__run_traffic_violation_detection):
			print("None of the car analytics modules have been setup, please setup and check output path again")
		else:
			print("THE OUTPUT FOR THIS RUN OF THE API WILL BE AT: " + self.__path_to_output)
	
	def path_to_alert_poller(self):
		return self.__path_to_output + "/"+ self.__start_time + "_alert_poller.txt"

	# Function Name: run
	# Description: based on flags from user input from __init__, sets flags in Tensorflow and runs the pipeline
	# Input: None
	def run(self):
		
		if not (self.__run_counting_cars or self.__run_detect_direction or self.__run_hscd or self.__run_traffic_violation_detection):
			print("please setup individual modules before running the pipeline")
			sys.exit(0)



		tf.reset_default_graph()

		FLAGS = argHandler()
		FLAGS.setDefaults()

		FLAGS.demo = self.__video_path  # video file to use, or if camera just put "camera"
		FLAGS.model = "darkflow/cfg/yolo.cfg"  # tensorflow model
		FLAGS.load = "darkflow/bin/yolo.weights"  # tensorflow weights
		FLAGS.threshold = 0.35  # threshold of decetion confidance (detection if confidance > threshold )
		FLAGS.gpu = 0  # how much of the GPU to use (between 0 and 1) 0 means use cpu
		FLAGS.track = True  # whether to activate tracking or not
		FLAGS.trackObj = "car"  # the object to be tracked
		FLAGS.saveVideo = True  # whether to save the video or not
		FLAGS.BK_MOG = False  # activate background substraction using cv2 MOG substraction,
		# to help in worst case scenarion when YOLO cannor predict(able to detect mouvement, it's not ideal but well)
		# helps only when number of detection < 5, as it is still better than no detection.
		# (NOTE : deep_sort only trained for people detection )
		FLAGS.tracker = "deep_sort"  # wich algorithm to use for tracking deep_sort/sort

		FLAGS.skip = 0  # how many frames to skipp between each detection to speed up the network
		FLAGS.csv = False  # whether to write csv file or not(only when tracking is set to True)
		FLAGS.display = True  # display the tracking or not

		# modules
		FLAGS.counting_cars = self.__run_counting_cars  # to enable counting cars application module
		FLAGS.direction_detection = self.__run_detect_direction  # run direction detection or skip
		FLAGS.speed_estimation = self.__run_hscd  # run speed estimation or skip
		FLAGS.traffic_signal_violation_detection = self.__run_traffic_violation_detection
		
		# FLAGS.application_dir = os.getcwd()
		# FLAGS.user_input_video_name = self.__user_input_video_name
		
		FLAGS.location_name = self.__location_name
		FLAGS.path_to_output = self.__path_to_output
		FLAGS.start_time = self.__start_time
		tfnet = TFNet(FLAGS)

		tfnet.camera()
		print("End of Demo.")
		
		# logger = open('../../../log/'+ self.__start_time+'_'+ self.__video_name+'_'+ '_log.txt', 'a')
		logger = open('../../../log/'+ 'log.txt', 'a')
		logger.write(self.__project_name +" was run for city: "+ self.__city_name+ " at: "+ self.__start_time +" with the following modules: ")
		
		if self.__run_hscd:
			logger.write(" High speed car detection ")

		if self.__run_counting_cars:
			logger.write(" Counting Cars ")

		if self.__run_detect_direction:
			logger.write(" Cars moving in wrong direction ")

		if self.__run_traffic_violation_detection:
			logger.write(" traffic signal violation ")

		logger.write("\n")

		# sys.exit(0)

	