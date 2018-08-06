import cv2
import numpy as np
import time
import datetime
import os
import json

class AbandonedObjectDetection(object):

	def __init__(self, video_path, city_name, location_name = " "):
		
		self.__video_path = video_path
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

		print("Done Init!")


	def detect_abandoned_objects(self):

		# capture video
		cap = cv2.VideoCapture(self.__video_path)
		
		# define video writer parameters
		out = None
		
		fps = 15.0
		fourcc = cv2.VideoWriter_fourcc(*'MPEG')

		# default algo variables
		interval = 0
		count = 0
		nPixel = 100
		flag = 0

		fgbg_mog2 = cv2.createBackgroundSubtractorMOG2(nPixel,cv2.THRESH_BINARY,2)
		remove_shadow = True

		# my variables
		aban = None
		sub = None

		# aban_eval_time_flag = None
		aban_interval = None

		algo_start_time = datetime.datetime.now().time().strftime('%H:%M:%S')

		current_date = str(datetime.datetime.now().date())
		# log_file = open(os.path.join(self.__path_to_log, current_date+".txt"), "a+")
		# log_file.write(algo_start_time + " Abandoend Object Detetecton algorithm started\n")
		# log_file.close()

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

			ret, frame = cap.read()
			cv2.imshow("original image", frame)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			# print("count: ", count)
			count += 1

			fgmask = fgbg_mog2.apply((gray))
			if remove_shadow:
				fgmask[fgmask == 127] = 0

			back = fgbg_mog2.getBackgroundImage()
			cv2.imshow('Static Background', back)
			#________________ bgs over _________

			if flag == 0:
				# initialize aban
				aban = cv2.absdiff(back, back)
				interval = 300
				flag = 10

			if ((flag == 10) and (count >= nPixel)):
				# count >= nPixel - meaning that the background has been successfully captured
				aban = back.copy()
				flag = 20

			start_time = datetime.datetime.strptime(algo_start_time, '%H:%M:%S')
			check_time = datetime.datetime.now().time().strftime('%H:%M:%S')
			end_time = datetime.datetime.strptime(check_time, '%H:%M:%S')
			diff = (end_time - start_time)
			# print("\nPrinting time: " +str((diff.seconds)))

			if(diff.seconds >= interval):
				aban = back.copy()
				algo_start_time = datetime.datetime.now().time().strftime('%H:%M:%S')

			
			
			# for video writing 
			# temp_aban = cv2.cvtColor(sub, cv2.COLOR_GRAY2BGR)
			# temp_concat = np.concatenate((frame, temp_aban), axis=1)
			# out.write(temp_concat)
			# cv2.imshow("output", temp_concat)

			
			if count >= nPixel:
				sub = cv2.absdiff(back, aban)
				# cv2.imshow("Abandoned Object", sub)
				
				ret, thresh1 = cv2.threshold(sub,127,255,cv2.THRESH_BINARY)
				kernel = np.ones((3,3),np.uint8)
				dilation = cv2.dilate(thresh1, kernel, iterations = 4)
				
				# further thresholding required to get a clearer distinction
				ret,thresh = cv2.threshold(dilation,127,255,0)
				im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
				
				# cv2.drawContours(frame, contours, -1, (0,255,0), 3)
				# cv2.imshow('dilation-cont', frame)
				test = frame.copy()
				if not contours:
					if out is not None:
						out.release()
						out = None

					# aban_eval_time_flag = True
					aban_interval = True
					
				else:

					if aban_interval:
						aban_counter = 0

					aban_interval = False
					aban_counter += 1

					# if an object is present for more than
					# threshold frames, then make a boinding box 
					# and start video writer
					if aban_counter > 150:

						if out == None:
							video_name = "abandoned_object_"+str(datetime.datetime.now().time().strftime('%H:%M:%S')) + ".avi"
							abs_video_path = os.path.join(dated_video_path, video_name)
							out = cv2.VideoWriter(abs_video_path, fourcc, 15.0, (frame.shape[1], frame.shape[0]))
							
							
							###### CHANGES WOULD BE REQUIRED HERE IN CASE OF HTTP END POINTS ######
							# make data json object
							data = {
								"module_name": self.__project_name,
								"location" : self.__location_name,
								"time" :  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
								"video_url" : abs_video_path
							}
							
							# dump json 
							json_file_name = "abandoned_object_"+str(datetime.datetime.now().time().strftime('%H:%M:%S')) + ".json"
							abs_json_file_path = os.path.join(dated_alert_path, json_file_name)
							with open(abs_json_file_path, 'w') as f:
								json.dump(data, f)

							log_file = open(os.path.join(self.__path_to_log, current_date+".txt"), "a+")
							log_file.write(abs_json_file_path)
							log_file.close()
	

							###### ########################################################### ######
						
						# merge contour logic remaining

						for cnt in contours:
							x,y,w,h = cv2.boundingRect(cnt)
							cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)

						# write video
						out.write(frame)
						

				cv2.imshow("Processed Output Window", frame)
				
			count += 1
			k = cv2.waitKey(1)
			if k == 27:
				break
				cap.release()

		cap.release()
			
		# log_file = open(os.path.join(self.__path_to_log, current_date+".txt"), "a+")
		# log_file.write(datetime.datetime.now().time().strftime('%H:%M:%S') + " Abandoend Object Detetecton algorithm ended\n")
		# log_file.close()
	




# put this in a jupyter notebook, Done!
video_path = '/media/anuj/Work-HDD/WORK/CLOUD-DRIVE/Google-Drive/Computer-Vision/Sample-Videos/Abandoned-Object/pets2006_1.avi'
vo = AbandonedObjectDetection(video_path, 'beijing', 'station')
vo.detect_abandoned_objects()


