#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import print_function

# vision specific imports
import numpy as np
import cv2

# general python util imports
import subprocess
import sys, glob, os, shutil
import re
from pprint import pprint
import configparser
from datetime import datetime
import time
import os
import psutil

"""
hard coding the extensions here as the software has been
tested out only for the mentioned extensions
"""
allowed_file_extensions = ['.mpg', '.webm', '.avi', '.mp4']

class VideoSynopsis(object):
	
	"""
	VideoSynopsis encapsulates functions for Video Synopsis and Reverese Object Search and other utils
	"""
	def __init__(self, video_path, city_name, location_name = " "):
		
		self.__video_path = video_path
		# self.__video_name = os.path.splitext(os.path.basename(video_path))[0]
		self.__video_name, self.__file_extension = os.path.splitext(os.path.basename(video_path))
		
		self.__video_length = "0 seconds"

		if not self.__file_extension in allowed_file_extensions:
			print("Invalid video file format, please check!")
			sys.exit(0)

		if not os.path.exists(video_path):
			print("\nVideo file does not exist. Please make sure path and video filename is proper.\n\n***ABORTING***\n")
			sys.exit(0)

		self.__numFrames, self.__frameHeight, self.__frameWidth, self.__fps = self.__video_props_util(video_path)

		
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
		self.__path_to_config = os.path.join(predef_dir_structure_path, 'config')

		
		# 2. path to config
		# path_to_config = '../../../config/config.ini'
		path_to_config = os.path.join(self.__path_to_config, "config.ini")
		
		if not os.path.exists(path_to_config):
			config = configparser.ConfigParser()
			config.optionxform = str
			config[self.__project_name +"_"+self.__city_name] = {}
			config[self.__project_name +"_"+self.__city_name]['user_output_path'] = " "


			with open(path_to_config, 'w') as configfile:
				config.write(configfile)

		else: 
			config = configparser.ConfigParser()
			config.optionxform = str
			config.read(path_to_config)

			if not config.has_section(self.__project_name +"_"+self.__city_name):
				config[self.__project_name +"_"+self.__city_name] = {}
				config[self.__project_name +"_"+self.__city_name]['user_output_path'] = " "	
				
				with open(path_to_config, 'w') as configfile:
					config.write(configfile)


		# Regex for checking if the path to video is rtsp
		pattern = re.compile(r'(rtsp://(.*))(\?.*)?')
		match = pattern.match(video_path)

		# if match is found, save the video as .avi
		if match:
			print("\nThis is an RTSP feed, saving video for offline processing!\n")
			self.__video_writer_util(video_path)
			print(self.__video_name + " has been saved at: ", end="")
			
			# data folder is where all our videos exists, if not present then create it
			if not os.path.exists("data"):
				os.makedirs("data")

			
			src_file_name = self.__video_name+".avi"
			destination = "./data/"
			self.__file_move_util(src_file_name, destination)

			# THIS IS A BIT WRONG
			# finally, instance variable for storing path to video
			self.__video_path = "./data/"+self.__video_name+".avi"
			print(self.__video_path)
			self.__numFrames, self.__frameHeight, self.__frameWidth, self.__fps = self.__video_props_util(self.__video_path)


		# if the path was a video file, then just store the path in the __video_path instance variable
		else:

			if not os.path.exists(video_path):
				print("\nVideo file does not exist. Please make sure path and video filename is proper.\n\n***ABORTING***\n")
				sys.exit(0)
			else:
				self.__video_path = video_path
		# self.__numFrames, self.__frameHeight, self.__frameWidth, self.__fps = self.__video_props_util(self.__video_path)

		if not os.path.exists('intermediate_output/'+self.__video_name):
			os.makedirs('intermediate_output/'+self.__video_name)

		
		
		config = configparser.ConfigParser()
		config.optionxform = str
		config.read(path_to_config)
		config_params = config[self.__project_name +"_"+self.__city_name]
		self.__user_output_path = str(config_params['user_output_path'])

		
		# its more of idiotic to ask user path to write
		# set output path
		# self.__user_output_path = self.__user_output_path.strip()
		
		# if not self.__user_output_path == "":
		# 	self.__path_to_output = self.__user_output_path + "output/" + self.__city_name +"/"+ self.__project_name + "/"+ self.__video_name
		# else:
		# 	self.__user_output_path = os.getcwd()
		# 	self.__path_to_output = self.__user_output_path + "/output/" + self.__city_name +"/"+ self.__project_name + "/"+ self.__video_name

		
		# create output directory if doesn't exists already
		if not os.path.exists(self.__path_to_output):
			os.makedirs(self.__path_to_output)


		self.__start_time = str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
		self.__unique_objects_obtained = False
		print("Done Init!")
	

	################################## UTIL FUNCTIONS #######################################
	
	def __video_props_util(self, video_path):
		cap = cv2.VideoCapture(video_path)
		numFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		fps = int(cap.get(cv2.CAP_PROP_FPS))
		cap.release()
		return numFrames, frameHeight, frameWidth, fps


	def __video_writer_util(self, video_path):
		cap = cv2.VideoCapture(video_path)
		if not cap.isOpened():
			print("error opening video file, exiting...")
			sys.exit(0)

		fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
		out = cv2.VideoWriter(self.__video_name+'.avi', fourcc, self.__fps, (self.__frameWidth, self.__frameHeight))

		for x in range(0, self.__numFrames-3):
			ret, frame = cap.read()
			out.write(frame)

		out.release()
		cap.release()

	
	def __file_move_util(self, src_file_name, destination):
		dest_file_name = destination+src_file_name
		if os.path.isfile(dest_file_name):
			os.remove(dest_file_name)

		if os.path.isfile(src_file_name):
			shutil.move(src_file_name, destination)	
	

	def __split_video_util(self):
		command = "python utils/video_to_frames/video_to_frames_helper.py " + self.__video_path
		subprocess.call(command, shell=True)

	
	def __run_detection_util(self):
		"""Runs YOLO darknet binary and detects objects frame-by-frame basis 

		"""
		path_to_video = os.path.join('./data', self.__video_name)
		
		if not os.path.exists(os.path.join(path_to_video, 'frames')):
			print("Split frames for given video not present.\ndata/" + self.__video_name + "/frames/ does not exist or has no image files within.\n")
			sys.exit(0)

		# delete existing detections file "det.txt"
		det_file_path = os.path.join(path_to_video, 'det')
		det_file = os.path.join(det_file_path, 'det.txt')
		if os.path.exists(det_file):
			os.remove(det_file)

		if not os.path.exists(det_file_path):
			os.makedirs(det_file_path)
		
		# write header to the detection file
		det_file = open(det_file, 'w')
		det_file.write('frame_number, object_id, x, y, w, h, conf, 3dx, 3dy, 3dz, object_name\n')
		det_file.close()
		
		# run YOLO darknet on all the frames
		frames_path = os.path.join(path_to_video, 'frames', '')
		run_yolo_command = "cd darknet && ./darknet detect cfg/yolo.cfg yolo.weights"
		subprocess.call(run_yolo_command + " ../" + frames_path, shell=True)


	def __generate_features_util(self):
		"""Generate features for traking using the detection output from darknet
		
		"""
		path_to_video = os.path.join('./data', self.__video_name)
		if not os.path.exists(os.path.join(path_to_video, 'frames')):
			print("Split frames for given <__video_name> folder not present.\ndata/" + self.__video_name + "/frames/ does not exist or has no image files within.\n")
			sys.exit(0)

		npy_file_path = os.path.join('deep_sort/resources/detections/', self.__video_name) 
		npy_file = os.path.join(npy_file_path, 'det.npy')
		if os.path.exists(npy_file):
			os.remove(npy_file)

		if not os.path.exists(npy_file_path):
			os.makedirs(npy_file_path)
		
		generate_feature_command = "cd deep_sort && python ../utils/deep_sort_tracking/deep_sort_tracking_helper.py --generate " + self.__video_name
		subprocess.call(generate_feature_command, shell=True)

	
	def __track_util(self):
		"""Run deep_sort tracking using tracking features 
		"""
		if not os.path.exists("data/" + self.__video_name + "/frames/"):
			print("Split frames for given <video_name> folder not present.\ndata/" + self.__video_name + "/frames/ does not exist or has no image files within.\n")
			sys.exit(0)

		if not os.path.exists("deep_sort/resources/detections/" + self.__video_name + "/det.npy"):
			print("det.npy for given <video_name> folder not present.\ndeep_sort/resources/detections/" + self.__video_name + " does not exist or has no det.npy files within.\n")
			sys.exit(0)

		track_command = "cd deep_sort && python ../utils/deep_sort_tracking/deep_sort_tracking_helper.py --track " + self.__video_name
		subprocess.call(track_command, shell=True)


	def get_video_length_util(self):
		video_length = ((self.__numFrames//self.__fps))
		if video_length < 60:
			self.__video_length = str(video_length) + " seconds"
			return str(video_length) + " seconds"
		elif (video_length > 60 and video_length <= 3450):
			self.__video_length = str(video_length/60) + " minutes"
			return str(video_length/60) + " minutes"
		elif (video_length > 3600):
			self.__video_length = str(video_length /3600) + " hours"
			return str(video_length /3600) + " hours" 


	def __time_in_seconds_util(self, time_length):
		time_in_seconds, time_metric = time_length.split(" ")

		if time_metric == 'seconds':
			time_in_seconds = int(time_in_seconds)*1
		elif time_metric == 'minutes':
			time_in_seconds = int(time_in_seconds)*60
		elif time_metric == 'hours':
			time_in_seconds = int(time_in_seconds)*60*60

		return time_in_seconds, time_metric


	def __check_time_split_util(self, input_time_length):
		video_time, video_time_metric = self.__time_in_seconds_util(self.__video_length)
		input_time, input_time_metric = self.__time_in_seconds_util(input_time_length)

		if (input_time <= video_time):
			return video_time, video_time_metric, input_time, input_time_metric
		else:
			return False, False, False, False


	def __get_object_list(self):
		"""Util function to get all the frames, all the objects in each frame, and their location
		as in every frame.

		Returns:
			all_frame_list (list): list of all the frames in a video in sequential order
			all_obj_list (list): list of all the objects 
			all_bbox_list (list): list of all the bounding boxes corresponding to the detected objects
		"""
		all_frame_list = []
		all_obj_list = []
		all_bbox_list = []

		path_to_det_file = "./data/" + self.__video_name + "/det/det.txt"
		with open(path_to_det_file) as f:
			next(f)
			for idx, line in enumerate(f):
				temp_list = [x for x in line.strip().split(",")]
				all_frame_list.append(int(temp_list[0]))
				all_obj_list.append(temp_list[-1])
				all_bbox_list.append(list(map(int, temp_list[2:-5])))
		return all_frame_list, all_obj_list, all_bbox_list


	def __obj_search_util(self, query_obj_name, all_obj_list, all_frame_list, all_bbox_list, path_to_frames):
		"""Util function for searching object using object name

		Args:
			query_obj_name (str): name of the object ONLY from the list of unique objects 
			as given by get_unique_objects function to the user.

		
		Writes a video to disc. The video contains the object searched for along with a bounding box over it
			
		"""
		bool_query_obj_loc = [x == query_obj_name for x in all_obj_list]
		frame_list = list(np.array(all_frame_list)[bool_query_obj_loc])
		bbox_list = np.array(all_bbox_list)[bool_query_obj_loc]
		
		# remove duplicate frames from list
		image_list = list(set(frame_list))
		image_list.sort()
		
		# frameWidth, frameHeight =  (self.__frameWidth, self.__frameHeight)
		fourcc = cv2.VideoWriter_fourcc(*'MJPG')
		video_string = self.__start_time+ "_" + query_obj_name+ "-" + self.__video_name + ".avi"
		out = cv2.VideoWriter(video_string, fourcc, self.__fps, (self.__frameWidth, self.__frameHeight))
		print("\n\nSearching the object: \"" + query_obj_name + "\" in " + self.__video_name + "....")
		print("\n\nPicking up the required frames and writing output to video file...")

		# used for array slicing
		for frame_idx in image_list:
			img = cv2.imread(path_to_frames + str(frame_idx) + ".jpg")
			indices = [i for i, x in enumerate(frame_list) if x == frame_idx]
			for boxes in bbox_list[indices]:
				x1 = boxes[0]
				y1 = boxes[1]
				w = boxes[2]
				h = boxes[3]
				x2 = x1 + w
				y2 = y1 + h
				cv2.rectangle(img, (x1, y1), (x2, y2),(0, 255, 0), 3)
			out.write(img)
		# print("Video has been written at: " + "./output/" +self.__video_name + "/"+video_string)
		print("Video has been written at: " + self.__path_to_output +"/"+video_string)
		
		# dest_file_name = "./output/"+self.__video_name+"/"+video_string
		# dest_file_name = self.__path_to_output + "/" + self.__video_name+"/"+video_string
		dest_file_name = self.__path_to_output + "/"+video_string
		if os.path.isfile(dest_file_name):
			os.remove(dest_file_name)

		src_file_name = video_string
		if os.path.isfile(src_file_name):
			# shutil.move(src_file_name, "./output/"+self.__video_name+"/")
			# shutil.move(src_file_name, self.__path_to_output+  "/"+ self.__video_name+"/")
			shutil.move(src_file_name, self.__path_to_output+  "/")

		return self.__path_to_output +"/"+video_string
		
	######################################### CORE API FUNCTIONS ##################################

	# @profile
	def preprocess(self):
		"""Core API function which does all pre-processing step by step
		   Prints the path to the preprocessed video
			
		"""
		start = time.time()
		print(" -- Preprocessing Started -- ")

		print(" \n -- 1. Splitting Video --> ", end="")
		# self.__split_video_util()
		print("COMPLETED")

		print(" \n -- 2. Runing Object Detection --> ", end="")
		# self.__run_detection_util()
		print("COMPLETED")

		print(" \n -- 3. Generating Features for Tracking --> ", end="")
		self.__generate_features_util()
		print("COMPLETED")

		# print(" \n -- 4. Performing Time Tagging  --> ", end="")
		# self.__track_util()
		# print("COMPLETED")

		# print(" \n -- Preprocessing Finished -- ")	
		
		# video_path = "intermediate_output/" + self.__video_name + "/timeTagged-" + self.__video_name + ".avi"
		# print("\nTime tagged video written at: ", video_path)
		# print("\nTOTAL LENGTH OF THE VIDEO IS: " + self.get_video_length_util())
		# end = time.time()

		# print("TOTAL PREPORCESS TIME: " + str(end-start))

		# pid = os.getpid()
		# py = psutil.Process(pid)
		# memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
		# print('MEMORY USED IN PREPORCESS: ' + str(memoryUse))

		
	def video_synopsis(self, time_split="", user_def_split=0):
		"""Core API function responsible for video synopsis
		
		Args:
			time_split (str): Takes in time in seconds or minutes or hours from the user.
			Note: A string is required, probably a UI input. Example: "10 seconds"
			
		"""
		start = time.time()
		time_split = time_split.strip()

		if time_split == "" and user_def_split == 0:
			print("video synopsis accepts time_split and user_def_split as parameters, please set EITHER of them")
			sys.exit(0)

		elif not time_split == "" and  not user_def_split == 0:
			print("set EITHER of them please")
			sys.exit(0)

		elif not time_split == "":
			num_splits = 0

			
			video_time, video_time_metric, input_time, input_time_metric = self.__check_time_split_util(time_split)
			if not (input_time and video_time):
				print("Invalid time split entered, please check and try again!")
				sys.exit(0)

			num_splits = video_time // input_time
			print("Number of splits determined: ", num_splits)
			print("Running video synopsis for ", time_split)

		elif not user_def_split == 0:
			num_splits = user_def_split

		

		
		video_path = "../../intermediate_output/"+ self.__video_name + "/timeTagged-" + self.__video_name + ".avi"
		video_merge_command = "cd utils/video_merge/ && python video_merge_helper.py " + video_path + " " + str(num_splits) + " " + str(self.__start_time)
		subprocess.call(video_merge_command, shell=True)

		# dest_file_name = "output/"+self.__video_name+ "/"+"merged-timeTagged-"+self.__video_name+".avi"
		# dest_file_name = self.__path_to_output+self.__video_name+ "/"+"merged-timeTagged-"+self.__video_name+".avi"
		dest_file_name = self.__path_to_output + "/"+ self.__start_time + "_merged-timeTagged-"+self.__video_name+".avi"
		if os.path.isfile(dest_file_name):
			os.remove(dest_file_name)

		src_file_name = "utils/video_merge/"+ self.__start_time + "_merged-timeTagged-"+self.__video_name+".avi"
		if os.path.isfile(src_file_name):
			# shutil.move(src_file_name, "output/"+self.__video_name+ "/")
			# shutil.move(src_file_name, self.__path_to_output+"/"+self.__video_name+ "/")
			shutil.move(src_file_name, self.__path_to_output+"/")

		# video_loc = self.__path_to_output+self.__video_name+"/"+"merged-timeTagged-"+self.__video_name+".avi"
		video_loc = self.__path_to_output+"/"+ self.__start_time + "_merged-timeTagged-"+self.__video_name+".avi"
		print("\nSynopsized video written at: ", video_loc)


		end = time.time()
		print("TOTAL SYSNOPSIS TIME: "+str(end-start))

		
		pid = os.getpid()
		py = psutil.Process(pid)
		memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
		# print('memory use:', memoryUse)
		print('MEMORY USED IN SYNOPSIS: ' + str(memoryUse))



		return video_loc
	
	def get_unique_objects(self):
		"""Core API function for getting the list of objects present in a given video sequence
	
		Prints list of unique objects.

		"""
		all_frame_list, all_obj_list, all_bbox_list = self.__get_object_list()
		unique_obj_list = list(set(all_obj_list))
		print("\n \nObject(s) present in the given video sequence: ")
		pprint(unique_obj_list)
		self.__unique_objects_obtained = True

	

	def reverse_object_search(self, query_obj_name=""):
		"""Core API function for reverese object search
		
		Wrapper function for searching object in video

		"""
		if not self.__unique_objects_obtained:
			print("first call get_unique_objects() function to check which object are present, then try again!")
			sys.exit(0)

		all_frame_list, all_obj_list, all_bbox_list = self.__get_object_list()
		unique_obj_list = list(set(all_obj_list))
		
		if not query_obj_name in unique_obj_list:
			print("Please mention the name of the object to be searched, cannot be an empty string!")
			sys.exit(0)

		path_to_frames = "./data/" + self.__video_name + "/frames/"
		return self.__obj_search_util(query_obj_name, all_obj_list, all_frame_list, all_bbox_list, path_to_frames)
		# print("TOTAL memory: " + str(%memit))




# if __name__ == "__main__":

	# obj = VideoSynopsis('data/red_light_sim_1.avi')
	# obj.get_video_length_util()
	# obj.video_synopsis("9 seconds")