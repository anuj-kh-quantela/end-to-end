import os
import sys

import configparser
import errno

from darkflow.darkflow.defaults import argHandler 
from darkflow.darkflow.net.build import TFNet

from applications.roi.direction_detection import draw_custom_roi_wd
from applications.roi.speed_estimation import draw_custom_roi_hscd
from applications.roi.traffic_signal_violation_detection import draw_custom_roi_tsvd

import tensorflow as tf
import glob
import shutil
from datetime import datetime
import cv2
import numpy as np
import shutil
import json


# Class Name: CarAnalytics
# Description: Class containing all flags to enable car analytics tasks
class CarAnalytics:
	# Function Name: __init__
	# Description: enables flags based on user input for car analytics tasks
	# Input: (video_path: path to input video)
	# def __init__(self, video_path, city_name, actual_name = " "):
	def __init__(self, video_path, city_name, location_name = " "):
		
		# 1. Define complete directory structure
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

		# 2. Config file settings
		# set path to config
		path_to_config = os.path.join(predef_dir_structure_path, 'config', 'config.ini')
		
		# make a config file when it doesn't exists
		if not os.path.exists(path_to_config):
			config = configparser.ConfigParser()
			config.optionxform = str
			config_name = self.__project_name +"_"+self.__city_name
			config[self.__project_name +"_"+self.__city_name] = {}
			config[self.__project_name +"_"+self.__city_name]['user_input_video_name'] = " "
			config[self.__project_name +"_"+self.__city_name]['user_output_path'] = " "
			config[self.__project_name +"_"+self.__city_name]['distance'] = "10"
			config[self.__project_name +"_"+self.__city_name]['frame_rate'] = "15"
			config[self.__project_name +"_"+self.__city_name]['speed_limit'] = "10"
			config[self.__project_name +"_"+self.__city_name]['roi_image_path'] = "/tmp/"

			with open(path_to_config, 'w') as configfile:
				config.write(configfile)

		# read config file and check for missing sections. Create sections, if missing
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

		# finally, read the config file and set necessary variables
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
		self.__video_name = os.path.splitext(os.path.basename(video_path))[0]
		self.__video_path = video_path


		# __user_output_path will be read from config. It is requrired when
		self.__user_output_path = self.__user_output_path.strip()
		
		if not self.__user_output_path == "":
			self.__user_output_path = os.path.join(self.__user_output_path, self.__city_name, self.__location_name, self.__project_name, 'output')
		else:
			self.__user_output_path = os.path.join(predef_dir_structure_path, 'output')

		
		self.__path_to_output = self.__user_output_path

		
		# create output directory if doesn't exists already
		if not os.path.exists(self.__path_to_output):
			os.makedirs(self.__path_to_output)

		
		self.__run_counting_cars = False
		self.__run_detect_direction = False
		self.__run_hscd = False
		self.__run_traffic_violation_detection = False
		self.__start_time = str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
		print('\n***Initialization Complete!***\n')
		print('\nPlease proceed for individual module setup\n')

		
		# Create a folder named video file in data folder
		self.__config_dir = os.path.join('./data', self.__video_name, 'config')
		print(self.__config_dir)

		# if not os.path.exists(self.__config_dir):
		#     try:
		#         os.makedirs(self.__config_dir)
		#     except OSError as e:
		#         if e.errno == errno.EEXIST:
		#             print("Warning: Directory already exists.\n")
		#         else:
		#             print("Can't create destination directory (%s)!" % self.__config_dir)
		#             sys.exit(0)

		if not os.path.exists(self.__config_dir):
			os.makedirs(self.__config_dir)

		config = configparser.ConfigParser()
		config.optionxform = str
		
		# # this doesn't gives any error even if the file doesn't exists! strange
		config_file_path = os.path.join(self.__config_dir, 'config.ini')
		config.read(config_file_path)


	# 1. counting cars module
	def setup_counting_cars(self):
		self.__run_counting_cars = True


	# 2. high speed car detection module 
	def setup_high_speed_detection(self, pDistance="", pFrameRate="", pSpeedLimit=""):
		if pDistance == "":
			pDistance = self.__distance
			
		if pFrameRate == "":
			pFrameRate = self.__frame_rate

		if pSpeedLimit == "":
			pSpeedLimit = self.__speed_limit

		draw_custom_roi_hscd.define_roi(self.__video_path, pDistance, pFrameRate, pSpeedLimit)
		self.__run_hscd = True
		

	def setup_wrong_direction_detection(self):
		draw_custom_roi_wd.define_roi(self.__video_path)
		self.__run_detect_direction = True

		
	def setup_traffic_signal_violation_detection(self):
		draw_custom_roi_tsvd.define_roi(self.__video_path, 'light')
		draw_custom_roi_tsvd.define_roi(self.__video_path, 'zone')
		self.__run_traffic_violation_detection = True
		self.__run_detect_direction = True
		
	
	def __show_output_path(self):
		if not (self.__run_counting_cars or self.__run_detect_direction or self.__run_hscd or self.__run_traffic_violation_detection):
			print("None of the car analytics modules have been setup, please setup and check output path again")
		else:
			print("THE OUTPUT FOR THIS RUN OF THE API WILL BE AT: " + self.__path_to_output)
	
	def __path_to_alert_poller(self):
		return self.__path_to_output + "/"+ self.__start_time + "_alert_poller.txt"

	# Function Name: run
	# Description: based on flags from user input from __init__, sets flags in Tensorflow and runs the pipeline
	# Input: None
	def run(self):
		
		if not (self.__run_counting_cars or self.__run_detect_direction or self.__run_hscd or self.__run_traffic_violation_detection):
			print("please setup individual modules before running the pipeline")
			sys.exit(0)

		print("OUTPUT WILL BE AT: " + str(self.__show_output_path()))
		print("ALERTS FOR THIS RUN WILL BE AT: " + str(self.__path_to_alert_poller()))

		tf.reset_default_graph()

		FLAGS = argHandler()
		FLAGS.setDefaults()

		FLAGS.demo = self.__video_path  # video file to use, or if camera just put "camera"
		FLAGS.model = "darkflow/cfg/yolo.cfg"  # tensorflow model
		FLAGS.load = "darkflow/bin/yolo.weights"  # tensorflow weights
		FLAGS.threshold = 0.35  # threshold of decetion confidance (detection if confidance > threshold )
		FLAGS.gpu = 0.85  # how much of the GPU to use (between 0 and 1) 0 means use cpu
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
		
		FLAGS.location_name = self.__location_name
		FLAGS.path_to_output = self.__path_to_output
		FLAGS.start_time = self.__start_time
		tfnet = TFNet(FLAGS)

		tfnet.camera()
		print("End of Demo.")
		