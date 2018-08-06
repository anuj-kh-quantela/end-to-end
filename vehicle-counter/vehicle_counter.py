import cv2,datetime,os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import IPython.display as Disp
import subprocess

import matplotlib

matplotlib.style.use('ggplot')

os.sys.path.append('roi-code/custom_roi/')
import draw_custom_roi
import datetime as dt
import math
import json

def data_resample(dataframe,rule='3T', method = 'sum', round_off=False, q=0.5):
    
    
		"""
		dataframe : time index dataframe
		rule : could be '3T' for 3 minute aggregation of 'H' for hourly aggregation
		method : 'sum'/'mean'
		*intially whole data is downsample to 1second by taking median value....
		-------parameter----------------
		Alias   Description
		B       business day frequency
		C       custom business day frequency (experimental)
		D       calendar day frequency
		W       weekly frequency
		M       month end frequency
		BM      business month end frequency
		CBM     custom business month end frequency
		MS      month start frequency
		BMS     business month start frequency
		CBMS    custom business month start frequency
		Q       quarter end frequency
		BQ      business quarter endfrequency
		QS      quarter start frequency
		BQS     business quarter start frequency
		A       year end frequency
		BA      business year end frequency
		AS      year start frequency
		BAS     business year start frequency
		BH      business hour frequency
		H       hourly frequency
		T, min  minutely frequency
		S       secondly frequency
		L, ms   milliseonds
		U, us   microseconds
		N       nanoseconds
		"""


		data_copy = dataframe.resample('S').median()
		data_copy.dropna(inplace=True)

		if round_off:
			if method == 'sum':
				return data_copy.resample(rule).sum().apply(np.ceil)
			if method == 'mean':
				return data_copy.resample(rule).mean().apply(np.ceil)
			if method == 'quantile':
				return data_copy.resample(rule).quantile(q).apply(np.ceil)

		else:

			if method == 'sum':
				return data_copy.resample(rule).sum()
			if method == 'mean':
				return data_copy.resample(rule).mean()
			if method == 'quantile':
				return data_copy.resample(rule).quantile(q)




class VehicleCounter(object):

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

	def order_points(self, pts):
		"""
		pts :np.array
		"""
		# initialzie a list of coordinates that will be ordered
		# such that the first entry in the list is the top-left,
		# the second entry is the top-right, the third is the
		# bottom-right, and the fourth is the bottom-left
		rect = np.zeros((4, 2), dtype = "float32")

		# the top-left point will have the smallest cv2sum, whereas
		# the bottom-right point will have the largest sum
		s = pts.sum(axis = 1)
		rect[0] = pts[np.argmin(s)]
		rect[2] = pts[np.argmax(s)]

		# now, compute the difference between the points, the
		# top-right point will have the smallest difference,
		# whereas the bottom-left will have the largest difference
		diff = np.diff(pts, axis = 1)
		rect[1] = pts[np.argmin(diff)]
		rect[3] = pts[np.argmax(diff)]

		# return the ordered coordinates
		return rect


	def four_point_transform(self, image, pts):
		# obtain a consistent order of the points and unpack them
		# individually
		#rect = order_points(pts)
		rect=pts
		(tl, tr, br, bl) = rect

		# compute the width of the new image, which will be the
		# maximum distance between bottom-right and bottom-left
		# x-coordiates or the top-right and top-left x-coordinates
		widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
		widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
		maxWidth = max(int(widthA), int(widthB))

		# compute the height of the new image, which will be the
		# maximum distance between the top-right and bottom-right
		# y-coordinates or the top-left and bottom-left y-coordinates
		heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
		heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
		maxHeight = max(int(heightA), int(heightB))

		# now that we have the dimensions of the new image, construct
		# the set of destination points to obtain a "birds eye view",
		# (i.e. top-down view) of the image, again specifying points
		# in the top-left, top-right, bottom-right, and bottom-left
		# order
		dst = np.array([
			[0, 0],
			[maxWidth - 1, 0],
			[maxWidth - 1, maxHeight - 1],
			[0, maxHeight - 1]], dtype = "float32")

		# compute the perspective transform matrix and then apply it
		M = cv2.getPerspectiveTransform(rect, dst)
		warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

		# return the warped image
		return warped


	def select_roi(self, frame):
		"""
		select roi in order 
		top-left, top-right, bottom-right, bottom-left
		"""
		copy_frame = frame.copy()
		test = draw_custom_roi.define_roi(frame, copy_frame)
		print("selected coordinates: ")
		print(test)
		return np.array(test,dtype=np.float32)
	

	def vehicle_counter(self, video_path=None, roi=None, plot_intermediate=False, minm_area = None ,remove_shadow=False, plot_matplot=False, check_interval=20, method='quantile', aggregation_time='3T', congestion_threshold=0.80):
		"""
		video_channel : video name or path of video or camera number
		roi : top-left, top-right, bottom-right, bottom-left
		plot_intermediate : plot intermediate plots
		min_area : minimum area that will be used to query
		remove_shadow : contours will  have no shadow
		
		"""

		if plot_intermediate:
			cv2.namedWindow('Actual Frames',cv2.WINDOW_NORMAL)
			# cv2.namedWindow('noise_removal',cv2.WINDOW_NORMAL)
			# cv2.namedWindow('non_moving_background',cv2.WINDOW_NORMAL)
			# cv2.namedWindow('detected_rectangle',cv2.WINDOW_NORMAL)
			# cv2.namedWindow('sure_background',cv2.WINDOW_NORMAL)
			# cv2.namedWindow('warped_im',cv2.WINDOW_NORMAL)
			cv2.namedWindow('Processed Output Window',cv2.WINDOW_NORMAL)
			
		fgbg_mog2 = cv2.createBackgroundSubtractorMOG2(200,cv2.THRESH_BINARY,1)
		fgbg_gmg = cv2.bgsegm.createBackgroundSubtractorGMG(200, cv2.THRESH_BINARY)
		
		cap = cv2.VideoCapture(self.__video_path)
		
		if roi is None:
			ret, frame = cap.read()
			pts = self.select_roi(frame)
		else: 
			pts =roi

		    


		#list of ration of sum(contours_area)/total_area
		congestion = []
		# distribution of areas based on histogram
		area_dist = []
		ret = True
		warped_im_area = None
		
		w,h = (cap.get(4),cap.get(3))
		
		text = 'select minimum area object \n skip frame : press s\nconfirmation : press c'
		label = None
		
		if minm_area is not None:
			# min_area_obj_type = str(raw_input("minimum object type? \nbike/car"))
			min_area_obj_type = "bike"

		algo_start_time = datetime.datetime.now().time().strftime('%H:%M:%S')

		while True:
			
			current_date = str(datetime.datetime.now().date())
			dated_output_path = os.path.join(self.__path_to_output, current_date)

			data_file_path = os.path.join(dated_output_path, 'data')
			images_file_path = os.path.join(dated_output_path, 'images')
			dated_alert_path = os.path.join(dated_output_path, 'alerts')

			# create output directory if doesn't exists already
			if not os.path.exists(data_file_path):
				os.makedirs(data_file_path)

			if not os.path.exists(images_file_path):
				os.makedirs(images_file_path)

			if not os.path.exists(dated_alert_path):
				os.makedirs(dated_alert_path)

			veh_cnt_file_path = os.path.join(data_file_path, 'vehicle_count.csv')
			congestion_file_path = os.path.join(data_file_path, 'congestion.csv')	




			try:
				time_1 = datetime.datetime.now()
				ret,frame = cap.read()
				if not (cap.isOpened() and ret):
					break
				if plot_intermediate:
					Disp.clear_output(wait=True)
				
				warped_im = self.four_point_transform(frame,pts)
				check_time = datetime.datetime.now().time().strftime('%H:%M:%S')
				start_time = dt.datetime.strptime(algo_start_time, '%H:%M:%S')
				end_time = dt.datetime.strptime(check_time, '%H:%M:%S')
				diff = (end_time - start_time)
				
					
				while minm_area is None:
					cv2.namedWindow('select_min_area',cv2.WINDOW_NORMAL)
					for idx,i in enumerate(text.split('\n')):
						cv2.putText(warped_im,i,(10,40*(idx+1)), cv2.FONT_HERSHEY_PLAIN, 1,(0,255,0),1,cv2.LINE_AA)


					k = cv2.waitKey(1)
					cv2.imshow("select_min_area",warped_im)
					if k==115:
						##  press s
						minm_area = None
						ret,frame = cap.read()
						warped_im = self.four_point_transform(frame,pts)
					if k==99:
						## press c
						bbox1 = cv2.selectROI('select_min_area', warped_im)
						min_area_obj_type = str(raw_input("minimum object type? \nbike/car"))
						minm_area = bbox1[2]*bbox1[3]
						cv2.destroyWindow('select_min_area')
						print("minimum area selected = "+str(minm_area))


				ret, frame = cap.read()
				
				warped_im = self.four_point_transform(frame, pts)
				if warped_im_area is  None:
					warped_im_area = warped_im.shape[0]*warped_im.shape[1]
				#warped_im=frame.copy()
				#plt.figure(figsize=(16,12))
				#plt.imshow(warped_im)
				#plt.show()

				#combining mmog2 and gmg
				fgmask = fgbg_mog2.apply((warped_im))
				if remove_shadow:
					fgmask[fgmask == 127] = 0
				fgmask_gmg = fgbg_gmg.apply((warped_im))

				fgmask=cv2.bitwise_and(fgmask,fgmask_gmg)

			
				# noise removal
				fgmask1=fgmask.copy()
				kernel = np.ones((3,3),np.uint8)
				opening = cv2.morphologyEx(fgmask1,cv2.MORPH_CROSS,kernel,1)
				
				# sure background area
				sure_bg = cv2.dilate(opening,kernel,iterations=4)
				#cv2.imshow('sure_background',sure_bg)
				#cv2.waitKey(0)

				im2, contours,hierarchy = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
				if len(contours)>0:
					areas=np.array(list(map(cv2.contourArea,contours)))
					args=np.argwhere(np.array(areas)>minm_area*.25).flatten()
					if len(args)>0:
						#There are multiple frames in  each second...thus each vehicle can be detected twice
						#hist=np.histogram(areas[args]/warped_im_area,bins=np.array(range(0,100,2),dtype=np.float32)/100)
						#hist = np.histogram(areas[args],bins=[minm_area,3*minm_area,6*minm_area,20*minm_area])

						#total area occupied on road
						#congestion.append([str(time_1) , areas[args].sum()/warped_im_area])
						fgmask_rect=fgmask.copy()
						areas_rect = []
						for arg in args:
							x,y,w,h = cv2.boundingRect(contours[arg])
							fgmask_rect=cv2.rectangle(fgmask_rect,(x,y),(x+w,y+h),(0,0,255), 2)
							cv2.rectangle(warped_im, (x,y),(x+w,y+h),(0,0,255), 2)

							areas_rect.append(w*h)
							
						if min_area_obj_type == "bike":
							label=['bike','car','minibus','bus','heavy_duty']
						if min_area_obj_type == "car":
							label=['car','minibus','bus','heavy_duty']

						if plot_matplot:
							# plt.figure(figsize=(12,6))
							# plt.subplot(121)
							# plt.plot(congestion)
							# plt.subplot(122)
							#op=plt.hist(areas[args]/warped_im_area)

							if  min_area_obj_type == 'car':
								hist_1 = plt.hist(areas_rect,\
												bins=[0,2*minm_area,4*minm_area,6*minm_area,10*minm_area])[0].tolist()
								plt.xticks([0,1*minm_area,3*minm_area,5*minm_area,8*minm_area],['','car','minibus','bus','heavy_duty'])
								plt.show()
							if min_area_obj_type == 'bike':

								hist_1 = plt.hist(areas_rect,\
												bins=[0,2*minm_area,4*minm_area,6*minm_area,8*minm_area,10*minm_area])[0].tolist()

								plt.xticks([0,1*minm_area,3*minm_area,5*minm_area,7*minm_area,9*minm_area],['','bike','car','minibus','bus','heavy_duty'])
								plt.show()
								# plt.legend([2*minm_area,4*minm_area,6*minm_area,10*minm_area])
								# # area distribution
								# area_dist.append([str(time_1)] + hist[0].tolist())
								# plt.show()
								
						else :
							if min_area_obj_type == 'car':
								hist_1 = np.histogram(areas_rect,bins=[0,2*minm_area,4*minm_area,6*minm_area,10*minm_area])[0].tolist()
							if min_area_obj_type == 'bike':
								hist_1 = np.histogram(areas_rect,bins=[0,2*minm_area,4*minm_area,6*minm_area,8*minm_area,10*minm_area])[0].tolist()


						

						
						veh_cnt = pd.DataFrame([[str(time_1)] + hist_1],columns = ['timestamp']+label)
						congestion_df =  pd.DataFrame([[str(time_1) , float(np.nansum(areas_rect))/warped_im_area]],columns=['timestamp','congestion'])
						
						# print(veh_cnt.tail())
						if not os.path.isfile(veh_cnt_file_path):
							veh_cnt.to_csv(veh_cnt_file_path, mode='a+',index=False)
						else:
							veh_cnt.to_csv(veh_cnt_file_path,mode='a+',index=False, header = False)

						
						if not os.path.isfile(congestion_file_path):
							congestion_df.to_csv(congestion_file_path, mode = 'a+',index=False)
						else:
							congestion_df.to_csv(congestion_file_path,mode = 'a+',index=False,header = False)

						
						# AGGREGATION LOGIC
						# read the values being dumped
						df = pd.read_csv(congestion_file_path)

						# convert index to date time index 
						df.timestamp = pd.DatetimeIndex(df.timestamp)

						# set the connverted column as the converted 
						df.set_index(['timestamp'], inplace=True, drop=True)
						
						# check every 'n' seconds
						if ((diff.seconds) % check_interval) == 0:
							
							
							congestion_value = data_resample(df.tail(10000), rule=aggregation_time, method=method, round_off=False).tolist()[0]
							if congestion_value > congestion_threshold:
								record_time = datetime.datetime.now().time().strftime('%H:%M:%S')
								
								image_name = "vehicle_counter_" + str(record_time) + ".jpg"
								abs_images_path = os.path.join(images_file_path, image_name)
								
								cv2.imwrite(abs_images_path, warped_im)

								veh_cnt_list = veh_cnt.values.tolist()[0]
								cdata = { "input": { "data":  { "SensorMetaInfo": { "CameraID" : "akashwani_east", "ServerId" : "vijaywada_PC_01", 	"ProductCategoryId" : "2", "FeatureId" : "9", "CameraDescription" : "cisco_cam_type_1", "LongDescription" : "high range camera", "Lx": "10.233N", "Ly": "20.98S" }, 
										"Event": { "EventID": 11, "EventDescription": "Threshold Congestion Exceeded" }, 
										"Data": {
											"CongressionThreshold" : str(congestion_threshold*100)+'%',
											"Vehicles" : {
												"BikeCount" : veh_cnt_list[1],
												"CarCount" : veh_cnt_list[2],
												"MiniBusCount" : veh_cnt_list[3],
												"HeavyVehicleCount" : veh_cnt_list[5],
												"AreaOccupied" : str(math.ceil(congestion_value*100))+'%',
												"ReportedTime" : "2018-02-26t10:23:51"
											},

											"CapturedTime" : veh_cnt_list[0],
											"ImageURL" : abs_images_path,
											"VideoURL" : "http://<ip-address>/video-analytics/hyderabad/akashwani_east/vehicle_counter/output/videos/2018_02_26_10_23_11_video.mp4"	
										}
										}
										},"configName": "TrafficCongestion", "groupName": "VideoAnalytics"
										}
								
								print(cdata)
								print("\n")
								# subprocess.call(["curl", "-X", "POST", "http://52.74.189.153:9090/api/v1/source/getInputData", "-H", "Cache-Control: no-cache", "-H", "Content-Type: application/json", "-H", "Postman-Token: 709d377f-a154-427a-83a1-25d14ce697f4", "-H", "x-access-token: M8U9LT3Q60GGUPGXN950NKFJ00A9E65Q", "-H", "tenantId:vijayawada.com", "-d", json.dumps(cdata)])

								json_file_name = "vehicle_counter_"+str(record_time) + ".json"
								abs_json_file_path = os.path.join(dated_alert_path, json_file_name)
								with open(abs_json_file_path, 'w') as f:
									json.dump(cdata, f)

					if plot_intermediate:
						cv2.imshow('Actual Frames', frame)    
						# cv2.imshow('sure_background',sure_bg)
						# cv2.imshow('non_moving_background',fgbg_mog2.getBackgroundImage())
						# cv2.imshow('warped_im',warped_im)
						cv2.imshow('Processed Output Window', warped_im)
						# cv2.imshow('detected_rectangle',fgmask_rect)
						# cv2.imshow('noise_removal',fgmask)
						
						k = cv2.waitKey(30) & 0xff
						if k == 27:
							break
					
				
			except Exception as e:
				print(e)
		cap.release()
		cv2.destroyAllWindows()



video_path = '/media/anuj/Work-HDD/WORK/CLOUD-DRIVE/Google-Drive/Computer-Vision/Sample-Videos/Car-Analytics/vijaywada_5min.mp4'
roi = np.array([(532, 256), (953, 307), (849, 808), (52, 541)],dtype = np.float32)
minm_area = 13455

vo = VehicleCounter(video_path, 'hyderabad', 'Deputy-01')
vo.vehicle_counter(minm_area=minm_area, roi=roi, plot_intermediate=True, check_interval=10, 
	method='quantile', aggregation_time='3T', congestion_threshold=0.20)


