import cv2,datetime,os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sys
import psutil
import time

class SlipAndFall(object):

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
	

	def detect_slipandfall(self, video_channel=None ,roi=None, debug=False, minm_area=None, maxm_area = None, remove_shadow=False, verbose=False):
		
		cap = cv2.VideoCapture(self.__video_path)

		if not cap.isOpened():
			print('Error opening video! Exiting.')
			sys.exit()

		if debug:
			cv2.namedWindow('person detected',cv2.WINDOW_NORMAL)
		
		cv2.namedWindow('Processed Output Window',cv2.WINDOW_NORMAL)
			
		fgbg_mog2 = cv2.createBackgroundSubtractorMOG2(200,cv2.THRESH_BINARY,1)
		fgbg_gmg = cv2.bgsegm.createBackgroundSubtractorGMG(200, cv2.THRESH_BINARY)
		
		

		ret = True
		w, h = (cap.get(4),cap.get(3))

		text = 'select minimum area object \n skip frame : press s\nconfirmation : press c'
		
		# minm_area = 8370	
		# maxm_area = 90360

		minm_area = minm_area
		maxm_area = maxm_area
		
		writer_flag = None
		res=[]
		out = None
		countdown_time = 0
		fourcc = cv2.VideoWriter_fourcc(*'XVID')

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
			

			try:
				ret,frame = cap.read()
				frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) 
				if not (cap.isOpened() and ret):
					break
		

				#------------------------------------minimum area selection---------------------------------------
				while minm_area is None:
					cv2.namedWindow('select_min_area',cv2.WINDOW_NORMAL)
					for idx,i in enumerate(text.split('\n')):
						cv2.putText(frame,i,(10,40*(idx+1)), cv2.FONT_HERSHEY_PLAIN, 1,(0,255,0),1,cv2.LINE_AA)


					k = cv2.waitKey(1)
					cv2.imshow("select_min_area",frame)
					if k==115:
						##  press s
						minm_area = None
						ret,frame = cap.read()
						frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
						
					if k==99:
						## press c
						bbox1 = cv2.selectROI('select_min_area', frame)
						#min_area_obj_type = str(raw_input("minimum object type? \nbike/car"))
						minm_area = bbox1[2]*bbox1[3]
						cv2.destroyWindow('select_min_area')
						print("minimum area selected = "+ str(minm_area))
				#------------------------------------------------------------------------------------------------
				

				#--------------------------------maximum area selection -----------------------------------------
				while maxm_area is None:
					cv2.namedWindow('select_maxm_area',cv2.WINDOW_NORMAL)
					for idx,i in enumerate(text.split('\n')):
						cv2.putText(frame,i,(10,40*(idx+1)), cv2.FONT_HERSHEY_PLAIN, 1,(0,255,0),1,cv2.LINE_AA)


					k = cv2.waitKey(1)
					cv2.imshow("select_maxm_area",frame)
					if k==115:
						##  press s
						minm_area = None
						ret,frame = cap.read()
						
					if k==99:
						## press c
						bbox1 = cv2.selectROI('select_maxm_area', frame)
						maxm_area = bbox1[2]*bbox1[3]
						cv2.destroyWindow('select_maxm_area')
						print("maximum area selected = "+ str(maxm_area))
						
				#-------------------------------------------------------------------------------------------------
				
				
				#combining mmog2 and gmg
				fgmask = fgbg_mog2.apply((frame))
				if remove_shadow:
					fgmask[fgmask == 127] = 0
				fgmask_gmg = fgbg_gmg.apply((frame))

				fgmask = cv2.bitwise_and(fgmask,fgmask_gmg)

		
				# noise removal
				fgmask1 = fgmask.copy()
				kernel = np.ones((3,3),np.uint8)
				opening = cv2.morphologyEx(fgmask1,cv2.MORPH_CROSS,kernel,1)
				
		
				im2, contours, hierarchy = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
				if len(contours)>0:
					areas = np.array(list(map(cv2.contourArea,contours)))
					args = np.argwhere(np.logical_and((maxm_area>np.array(areas)), np.array(areas)>0.33*minm_area)).flatten()
					fgmask_rect = fgmask
					
					if len(args)>0:
						ratio = 0
						fgmask_rect = fgmask.copy()
						for arg in args:
							x,y,w,h = cv2.boundingRect(contours[arg])
							cv2.rectangle(fgmask_rect,(x,y),(x+w,y+h),(255,255,255),2)
							
							# SLIP AND FALL LOGIC
							# change in the aspect ratio
							# along with change in the angle 

							try:
								ratio = float(w)/h
							except Exception as e:
								print e

							if  ratio > 1.0:

								# put rectangle when slip and fall is detected
								cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,255),2)
								cv2.putText(frame, "Slip and Fall Detected",(40,40), cv2.FONT_HERSHEY_PLAIN, 2,(0, 0, 255),2, cv2.LINE_AA)
								if ((writer_flag is None) and countdown_time == 0):
									writer_flag = 1
									video_name = "slip_and_fall_"+str(datetime.datetime.now().time().strftime('%H:%M:%S')) + ".avi"
									abs_video_path = os.path.join(dated_video_path, video_name)
									out = cv2.VideoWriter(abs_video_path, fourcc, 15.0, (frame.shape[1], frame.shape[0]))
									
								
								# WRITTEN AT WRONG PLACE
								# THIS IS FOR TIGHTER THRESHOLD
								# theta = (np.arctan(ratio))
								# res.append([ratio,theta])
								# print([ratio,theta])
								countdown_time = 60
								out.write(frame)

							else:
								writer_flag = None
								if out is not None:
									countdown_time -= 1
									cv2.putText(frame, "Slip and Fall being captured",(40,40), cv2.FONT_HERSHEY_PLAIN, 2,(0, 0, 255),2, cv2.LINE_AA)
									cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,255),2)
									out.write(frame)
									if countdown_time == 0:
										out.release()
										out = None

					if debug:
						cv2.imshow('person detected', fgmask_rect)

					cv2.imshow('Processed Output Window', frame)    
					k = cv2.waitKey(1) & 0xff
					if k == 27:
						break
				
			except Exception as e:
				print(e)
				cap.release()
				cv2.destroyAllWindows()


video_path = '/media/anuj/Work-HDD/WORK/CLOUD-DRIVE/Google-Drive/Computer-Vision/Sample-Videos/Slip-And-Fall/test-4.webm'
vo = SlipAndFall(video_path, 'test-city', 'test-location')
vo.detect_slipandfall(debug=False,remove_shadow=True, minm_area=8370, maxm_area=90360)

