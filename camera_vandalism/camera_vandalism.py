import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import cv2,itertools,os,pickle,time
import sched, datetime
import warnings
#warnings.filterwarnings("ignore")
import scipy as sp
import IPython.display as Disp

import errno
from socket import error as socket_error
import cv2
import numpy as np
import socket
import time
import re
from urlparse import urlparse
import sys
import configparser
import ast
from datetime import datetime
import psutil
import time
import datetime as dt


allowed_file_extensions = ['.mpg', '.webm', '.mp4', '.mkv', '.avi']

def diff_GDD(I1,I2,num_bins=1,verbose=False):
	"""
	I1 : image 1
	I2 : image 2
	num_bins : internal computation for calculating 2d histogram
	verbose : print internally computed value
	"""    
	l=[]
	for i in [I1,I2]:
		gray_image = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
		sobelx = cv2.Sobel(gray_image,cv2.CV_64F,1,0,ksize=5)
		sobely = cv2.Sobel(gray_image,cv2.CV_64F,0,1,ksize=5)
	   # magn=np.sqrt(sobelx*sobelx+sobely*sobely)
		grad=np.arctan2(sobely,sobelx)
		bin_i=num_bins*(grad+np.pi*.5)/np.pi
		l.append(bin_i)
	diff1=(abs((l[1]-l[0]))).sum()
	if verbose:
		print(diff1)
	return diff1
	
	


def diff_L1R(I1, I2,num_bins=1,verbose=False):
	"""
	I1 : image 1
	I2 : image 2
	num_bins : internal computation for calculating 2d histogram
	verbose : print internally computed value
	""" 
	l=[]
	for i in [I1,I2]:
		bin_l1=(num_bins*i.mean(axis=2)).flatten()
		bin_g=(num_bins*(i.max(axis=2)-i.min(axis=2))).flatten()
		H, xedges, yedges=np.histogram2d(bin_l1,bin_g,bins=10,range=[[0,255],[0,255]])
		l.append(H)
	diff1=(abs(l[0]-l[1])).sum()   
	if verbose:
		print(diff1)
	return diff1


def diff_L1R_1(i):
	diff_L1R_1.__name__='diff_L1R_1'
	return diff_L1R(i[0],i[1])


def diff_GDD_1(i):
	diff_GDD_1.__name__='diff_GDD_1'
	return diff_GDD(i[0],i[1])


class CameraVandalism(object):

	def __init__(self, video_path, city_name, location_name = " "):

		# video_path = video_path.strip()
		self.__is_rtsp = False
		self.__is_video = False
		self.__is_camera = False

		if video_path == ''  or video_path == ' ' or video_path == None:
			print("Please provide some value")
			sys.exit(0)

		elif type(video_path) == int:
			print("== Using locally connected camera ==")
			self.__video_name = "camera_location"
			self.__is_camera = True

		elif type(video_path) == str:

			self.__video_name, self.__file_extension = os.path.splitext(os.path.basename(video_path))
			
			# print self.__video_name, self.__file_extension
			if not self.__file_extension in allowed_file_extensions:
				print("Invalid video file format, please check!")
				sys.exit(0)

			# Regex for checking if the path to video is rtsp
			pattern = re.compile(r'(rtsp://(.*))(\?.*)?')
			match = pattern.match(video_path)

			
			if match:
				
				self.__rtsp_url = urlparse(video_path)
				self.__is_video = False
				self.__is_rtsp = True
				print("== Using RTSP feed ==")

			else:
				if not os.path.exists(video_path):
					print("\nVideo file does not exist. Please make sure path and video filename is proper.\n\n***ABORTING***\n")
					sys.exit(0)
				
				self.__video_path = video_path
				self.__is_video = True
				self.__is_rtsp = False
				print("== Using locally stored video file ==")
				

		cap = cv2.VideoCapture(video_path)
		self.__fps = int(cap.get(cv2.CAP_PROP_FPS))
		self.__frame_width = int(cap.get(3))
		self.__frame_height = int(cap.get(4))
		cap.release()

		self.__quit_process = False

		
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


		path_to_config = os.path.join(self.__path_to_config, "camera_vandalism.ini") 
		
		if not os.path.exists(path_to_config):
			config = configparser.ConfigParser()
			config.optionxform = str
			config[self.__project_name +"_"+self.__city_name] = {}
			config[self.__project_name +"_"+self.__city_name]['user_output_path'] = " "
			
			config[self.__project_name +"_"+self.__city_name]['size_short_pool'] = "10"
			config[self.__project_name +"_"+self.__city_name]['n_jobs'] = "8"
			config[self.__project_name +"_"+self.__city_name]['plot'] = "False"
			config[self.__project_name +"_"+self.__city_name]['file_name'] = "test_cam_"
			config[self.__project_name +"_"+self.__city_name]['thresh_shift_param'] = "3"
			config[self.__project_name +"_"+self.__city_name]['eps_frame'] = "20"
			config[self.__project_name +"_"+self.__city_name]['frame_to_start'] = "0"
			config[self.__project_name +"_"+self.__city_name]['resize_ratio'] = "0"
			config[self.__project_name +"_"+self.__city_name]['skip_frames'] = "0"
				

			with open(path_to_config, 'w') as configfile:
				config.write(configfile)

		else: 
			config = configparser.ConfigParser()
			config.optionxform = str
			config.read(path_to_config)

			if not config.has_section(self.__project_name +"_"+self.__city_name):
				config[self.__project_name +"_"+self.__city_name] = {}
				config[self.__project_name +"_"+self.__city_name]['user_output_path'] = " "

				config[self.__project_name +"_"+self.__city_name]['size_short_pool'] = "10"
				config[self.__project_name +"_"+self.__city_name]['n_jobs'] = "8"
				config[self.__project_name +"_"+self.__city_name]['plot'] = "False"
				config[self.__project_name +"_"+self.__city_name]['file_name'] = "test_cam_"
				config[self.__project_name +"_"+self.__city_name]['thresh_shift_param'] = "3"
				config[self.__project_name +"_"+self.__city_name]['eps_frame'] = "20"
				config[self.__project_name +"_"+self.__city_name]['frame_to_start'] = "0"
				config[self.__project_name +"_"+self.__city_name]['resize_ratio'] = "0"
				config[self.__project_name +"_"+self.__city_name]['skip_frames'] = "0"

				with open(path_to_config, 'w') as configfile:
					config.write(configfile)


		config = configparser.ConfigParser()
		config.optionxform = str
		config.read(path_to_config)
		config_params = config[self.__project_name +"_"+self.__city_name]
		# self.__user_input_video_name = str(config_params['user_input_video_name'])
		
		# 1.
		self.__user_output_path = str(config_params['user_output_path'])
		
		self.__size_short_pool = int(config_params['size_short_pool'])
		self.__n_jobs = int(config_params['n_jobs'])
		self.__plot = ast.literal_eval(config_params['plot'])
		self.__file_name = str(config_params['file_name'])
		self.__thresh_shift_param = int(config_params['thresh_shift_param'])
		self.__eps_frame = int(config_params['eps_frame'])
		self.__frame_to_start = int(config_params['frame_to_start'])
		self.__resize_ratio = int(config_params['resize_ratio'])
		self.__skip_frames = int(config_params['skip_frames'])


		if self.__file_name == '':
			if self.__is_video:
				self.__file_name = 'default_video_file_'
			if self.__is_rtsp:
				self.__file_name = 'default_rtsp_file_'
			if self.__is_camera:
				self.__file_name = 'default_camera_file_'
		
		self.__intermediate_file_path = os.path.join(predef_dir_structure_path, 'intermediate_files')
		

		# create output directory if doesn't exists already
		if not os.path.exists(self.__path_to_output):
			os.makedirs(self.__path_to_output)

		
		if not os.path.exists(self.__intermediate_file_path):
			os.makedirs(self.__intermediate_file_path)
		
		
		self.__start_time = str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

		print("Done Init!")
		print('\n***Initialization Complete!***\n')
		


		#----------------------- long term processing for camera------------------
	def create_long_term_camera(self, file_name,n_jobs=3,skip_secs=1,channel=0,f=diff_L1R_1,mask_background=False, plot=False):
		"""
		file_name : prefix of files that are going to be get stored as 
		n_jobs : parallel threads
		skip_secs : second to skip between each computation
		f : gradient  computation method
		mask_background : if you want to mask background(experimental) 
		
		
		"""

		# file_name = self.__intermediate_file_path + "/" + file_name

		if ((os.path.isfile(file_name+'long_term_frame.pickle')) &(os.path.isfile(file_name+'d_long.pickle'))):
			print('loading from existing file')
			d_long=pickle.load(open(file_name+'d_long.pickle','rb'))
			frame_list=pickle.load(open(file_name+'long_term_frame.pickle','rb'))
		else:

			if n_jobs>1:
				p=Pool(n_jobs)
			#ret,frame=cap.read()
			frame_list=[]
			d_long=[]
			#cap=cv2.VideoCapture(0)
			cap=cv2.VideoCapture(channel)
			fps=cap.get(5)
			fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
			while cap.isOpened():
				t1=datetime.now()
				print('getting new frame')
				ret,frame=cap.read()
				if mask_background:
					#fgmask = fgbg.apply(frame,1)
					fgmask = fgbg.apply(frame,.5)
					frame = cv2.bitwise_and(frame,frame,mask = fgmask)
				frame_list.append(frame)
				images_comb=list(itertools.product(frame_list,[frame]))
				
				if plot:
					plt.imshow(frame)
					plt.show()
	#             cv2.imshow("preview long term",frame)
	#             cv2.waitKey(0)
				try:
					print('processing function')
					if n_jobs>1:
						d_long.append(p.map(f,images_comb))
					else:
						d_long.append(map(f,images_comb))
						
				except Exception as e:
					print(e)
					print("no mapping generated")
					d_long.append([np.nan])        
				Disp.clear_output(wait=True)

				if len(frame_list)==60:
					## require edit later
					break

				t2=datetime.now()
				time.sleep(max(0,skip_secs-(t2-t1).total_seconds()))    
			pickle.dump(frame_list,open(file_name+'long_term_frame.pickle','wb'))
			pickle.dump(d_long,open(file_name+'d_long.pickle','wb'))
			if n_jobs>1:
				p.close()
			cap.release()
			cv2.destroyAllWindows()
		return frame_list,d_long

	

	#-------------------test-Camera-----------------------------------------
	def test_on_camera(self, channel=0,size_short_pool=10, f=diff_GDD_1,n_jobs=4,plot=False,file_name='bang_office_5_camera_testing_',thresh_shift_param=3,eps_frame=20):
		"""
		channel : videochannel
		size_short_pool : small the pool size faster the detection is, but small size of pool will give more false results as well
		f : evaluating fuction diff_GDD_1/ diff_HCD_1 / diff_L1R_1
		n_jobs : number of parallel jobs to be used
		plot : tracking progress plot
		file_name : prefix of all internal files to be stored(IMPORTANT : for historical data if file is available then it will be used directly )
		thresh_shift_param : numerical parameter to shift threshold, its a multiplicative param, which stretch threshold 
		eps_frames : eps_frame number of frame that should be considered to decide fix threshold 
		thresh_l = median(x,-0)-eps*thresh_shift_param
		thresh_u = median(x,90)+eps*thresh_shift_param
		"""
		#--------------------------------------------------
		font                   = cv2.FONT_HERSHEY_SIMPLEX
		topLeft                = (10, 25)
		fontScale              = .8
		fontColor              = (255,255,255)
		lineType               = 2
		#---------------------------------------------------
		# initial inputs
		channel=channel #either it is 0 or 1
		size_short_pool=size_short_pool
		#skip_frames=1
		d_between=[]
		norm_diff_li=[]
		norm_diff_li_1_good=[]
		f=f
		n_jobs=n_jobs
		
		write_video = True
		out_video = None
		fps = 2
		file_name = self.__intermediate_file_path + "/" + file_name

		# directory_path=os.path.dirname(file_name)
		#---------------------------------------------------
		vandalised=False
		vandal_start=True
		rest_down_period=0
		eps=None
		thresh_shift_param=thresh_shift_param
		#cap=cv2.VideoCapture(1)
		#p=Pool(3)
		#--------------------------------------------------
		print(size_short_pool, thresh_shift_param, eps)
		try:
			#frame_list,d_long=create_long_term_camera(datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")+'_camera_testing_'+f.__name__+'_',f=f,skip_secs=1,n_jobs=1)
			frame_list,d_long=self.create_long_term_camera(file_name+f.__name__+'_',f=f,skip_secs=2,channel=channel,n_jobs=n_jobs, plot=plot)
			print("file loaded")
			p=Pool(n_jobs)
			
			cv2.namedWindow('preview',cv2.WINDOW_NORMAL)
			cap=cv2.VideoCapture(channel)

			#frame_to_start=fps
			#cap.set(1,frame_to_start)
			#d_long=pickle.load(open('iter_1_d_long.pickle'))
			#frame_list=pickle.load(open('iter_1_long_term_frame.pickle'))
			d_long_df=pd.DataFrame(d_long)
			for i in range(d_long_df.shape[0]):
				d_long_df.iloc[i,i]=np.nan
			#--------------------------------------------
			#testing on camera
			#cap=cv2.VideoCapture(0)            
			#cap.set(cv2.CAP_PROP_FRAME_COUNT,1)
			#print('Frame returned=',ret)
			d_long_med=np.nanmedian(d_long_df.values)
			print("cap isopened=",cap.isOpened())


			

			while cap.isOpened():
				
			#for frame in frames:
				ret,frame=cap.read()
				#print('running frame',frame_to_start)
				print("creating combination pool")
				images_comb=list(itertools.product(frame_list,[frame]))
				print("combination pool created \n evaluting gradient for each combination ")
				d_between.append(p.map(f,images_comb))
				Disp.clear_output(wait=True)
				norm_diff=np.log10((np.nanmedian(pd.DataFrame(d_between)))/d_long_med)
				norm_diff_li.append(norm_diff)
				#plt.subplot(132), plt.plot(norm_diff_li)

				#print('normalized_diff=',norm_diff)
				#frame_to_start+=skip_frames

				#ret,frame=cap.read()
		#             for i in range(skip_frames+1):
		#                 ret,frame=cap.read()
				
				if len(d_between)>=size_short_pool:
					# print("resizing pool")
					d_between=d_between[1:]
					#--------------------fixing size of norm_diff_li_1 list as 500----------------------
					norm_diff_li=norm_diff_li[-500:]
					norm_diff_li_1=np.diff(norm_diff_li)
					norm_diff_1=norm_diff_li_1[-1]
					if eps is None:
						if len(norm_diff_li)<=eps_frame:
							eps=max(abs(norm_diff_li_1))
					#thresh=runningMedian(norm_diff_li_1,int(size_short_pool))[-2]
					if not vandalised:
						norm_diff_li_1_good.append(norm_diff_li_1[-1])
						thresh_l=np.percentile(norm_diff_li_1_good[-thresh_shift_param*size_short_pool:],q=10)-thresh_shift_param*eps
						thresh_u=np.percentile(norm_diff_li_1_good[-thresh_shift_param*size_short_pool:],q=90)+thresh_shift_param*eps
					
					rest_down_period-=1
					# print(norm_diff_1,eps,thresh_l,thresh_u)
					#if ((norm_diff_1>max(.005,thresh*1.1)) & (rest_down_period<=0)) :

					if out_video is not None:
						out_video.write(frame)

					if ((norm_diff_1>thresh_u) & (rest_down_period<=0) & (len(norm_diff_li)>=eps_frame)) :
						vandalised=True
						cv2.putText(frame,'vandalism Started',topLeft, font, fontScale,(0,0,255),lineType)
						fp = open('alert.txt', 'a')
						fp.write('Event: vandalism started at: ' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + '\n')
						fp.close()

						if write_video:
							video_save_string = self.__path_to_output +"/"+ str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + "_" + self.__video_name + ".avi"
							out_video = cv2.VideoWriter(video_save_string, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (self.__frame_width, self.__frame_height))
							write_video = False

						#print("vandalism started")
					#elif norm_diff_1<(-.01):qq
					elif ((norm_diff_1<(thresh_l))& (vandalised)):
						fp = open('alert.txt', 'a')
						fp.write('Event: vandalism removed at: ' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + '\n')
						fp.close()
						#print("vandalism removed")
						cv2.putText(frame,'vandalism removed',topLeft, font, fontScale,(0,255,255),lineType)
						vandalised=False
						rest_down_period=size_short_pool

						if out_video is not None:
							out_video.release()
							out_video = None
							write_video = True

					elif not vandalised:  
						#print("no vandalism")
						for i in range(cv2.CAP_PROP_FRAME_COUNT-1):
							ret,frame=cap.read()
						cv2.putText(frame,'no vandalism',topLeft, font, fontScale,(0,255,0),lineType)

					else:

						fp = open('alert.txt', 'a')
						fp.write('Event: vandalism started but not removed' + '\n')
						fp.close()

						#print("vandalism started but not removed")
						cv2.putText(frame,'vandalism started but not removed',topLeft, font, fontScale,(255,0,0),lineType)
		#                     u=np.percentile(norm_diff_li_1,q=80)
		#                     m=l=np.percentile(norm_diff_li_1,q=50)
		#                     l=np.percentile(norm_diff_li_1,q=20)
		#                     norm_diff_li_1[((norm_diff_li_1>l)&(norm_diff_li_1<u))]=m
					if plot:
						fig=plt.figure(figsize=(18, 4))
						plt.subplot(131), plt.imshow(frame.astype(np.uint8))
						plt.subplot(132),plt.plot(norm_diff_li)
						plt.subplot(133), plt.plot(norm_diff_li_1)
						plt.show()
					
					# print(thresh_l,thresh_u,eps)
					cv2.imshow("preview",frame)
					cv2.waitKey(1)

					 

				if  (cv2.waitKey(1) & 0xFF == ord('q')):
					break

			p.close()
			cap.release()
			cv2.destroyAllWindows()  
		except Exception as e:
			print(e)

		cv2.destroyAllWindows()



	#----------------------------------------------long term preprocessing for video----------------
	def create_long_term_vid(self, cap,file_name,n_jobs=3,skip_secs=1,f=diff_L1R_1,resize_ratio=.5,plot=False):
		"""
		cap : videocapture instance
		file_name : prefix of files that are going to be get stored as 
		n_jobs : parallel threads
		skip_secs : second to skip between each computation
		f : gradient  computation method
		resize_ratio : resize ratio for large images
		plot : plot intermediate plot
		"""
		if ((os.path.isfile(file_name+'long_term_frame.pickle')) &(os.path.isfile(file_name+'d_long.pickle'))):
			print('loading from existing file')
			d_long=pickle.load(open(file_name+'d_long.pickle','rb'))
			frame_list=pickle.load(open(file_name+'long_term_frame.pickle','rb'))
		else:
			skip_secs=skip_secs
			p=Pool(n_jobs)
			ret,frame=cap.read()
			if not ret:
				ret,frame=cap.read(self.video_name)
			frame_list=[]
			d_long=[]
			fps=cap.get(5)
			#skip first few seconds data *
			frame_number=fps*0
			while ret:
				print('reading frame',ret)
				ret,frame=cap.read()
				if resize_ratio>0:
					frame1=frame.copy()
					frame = cv2.resize(frame1, (0,0), fx=resize_ratio, fy=resize_ratio)


				frame_list.append(frame)
				images_comb=list(itertools.product(frame_list,[frame]))
				if plot:
					plt.imshow(frame.astype(np.uint8))
					plt.show()
				try:
					d_long.append(p.map(f,images_comb))
				except:
					d_long=d_long+[np.nan]
				
				frame_number+=1+int(fps*skip_secs)
				cap.set(1,frame_number)
				if plot:
					Disp.clear_output(wait=True)
				if len(frame_list)==60:
					## require edit later
					break
			print("last frame picked",frame_number)
			pickle.dump(frame_list,open(file_name+'long_term_frame.pickle','wb'))
			pickle.dump(d_long,open(file_name+'d_long.pickle','wb'))
			p.close()
		return frame_list,d_long



	#---------------------------- test- Video------------------------------------
	def test_on_video(self, input_video,f=diff_GDD_1,size_short_pool=10,n_jobs=7,file_name='bangalore_test_1_',plot=False,thresh_shift_param=3,eps_frame=20,frame_to_start=0,resize_ratio=.5,skip_frames=3):
		"""
		input_video : path of input video
		size_short_pool : small the pool size faster the detection is, but small size of pool will give more false results as well
		f : evaluating fuction diff_GDD_1/ diff_HCD_1 / diff_L1R_1
		n_jobs : number of parallel jobs to be used
		plot : tracking progress plot
		file_name : prefix of all internal files to be stored(IMPORTANT : for historical data if file is available then it will be used directly )
		thresh_shift_param : numerical parameter to shift threshold, its a multiplicative param, which stretch threshold 
		eps_frames : eps_frame number of frame that should be considered to decide fix threshold 
		frame_to_start : frame at which prediction should be started (as in video prediction, a lot of time can be for non-vandalised period )
		thresh_l = median(x,-0)-eps*thresh_shift_param
		thresh_u = median(x,90)+eps*thresh_shift_param
		"""
		#--------------------------------------------------
		font                   = cv2.FONT_HERSHEY_SIMPLEX
		topLeft                = (10, 25)
		fontScale              = .8
		fontColor              = (255,255,255)
		lineType               = 2
		#---------------------------------------------------
		video_name=input_video
		f=f
		n_jobs=n_jobs
		file_name=file_name
		size_short_pool=size_short_pool
		cap=cv2.VideoCapture(video_name)
		thresh_shift_param=thresh_shift_param
		eps_frame=eps_frame 
		frame_to_start=frame_to_start
		resize_ratio=resize_ratio
		#---------------------------------------------------
		vandalised=False
		vandal_start=True
		rest_down_period=0
		eps=None

		skip_frames=skip_frames

		write_video = True
		out_video = None
		fps = 2
		file_name = self.__intermediate_file_path + "/" + file_name
		fps = 15.0
		fourcc = cv2.VideoWriter_fourcc(*'MPEG')

		#cap=cv2.VideoCapture(1)
		#p=Pool(3)
		#--------------------------------------------------
		#frame to skip in each short pool
		d_between=[]
		norm_diff_li=[]
		norm_diff_li_1_good=[]
		p=Pool(n_jobs)
		f=f
		frame_idx=0
		vandalised=False
		rest_down_period=0

		start = time.time()
		
		try:
			frame_list,d_long=self.create_long_term_vid(cap,file_name+f.__name__+'_',n_jobs=n_jobs, f=f,skip_secs=0,resize_ratio=resize_ratio)
	#         for i in range(frame_to_start-1):
	#             ret,frame1=cap.read()
			cap.set(1,frame_to_start)
			#d_long=pickle.load(open('iter_1_d_long.pickle'))
			#frame_list=pickle.load(open('iter_1_long_term_frame.pickle'))
			d_long_df=pd.DataFrame(d_long)
			for i in range(d_long_df.shape[0]):
				d_long_df.iloc[i,i]=np.nan
			
			ret,frame=cap.read()
			if resize_ratio>0:
				frame1=frame.copy()
				frame = cv2.resize(frame1, (0,0), fx=resize_ratio, fy=resize_ratio)

			d_long_med=np.nanmedian(d_long_df.values)
			while ret:

				###########################################
				# very important setting for writing output 

				current_date = str(dt.datetime.now().date())
				dated_output_path = os.path.join(self.__path_to_output, current_date)

				dated_video_path = os.path.join(dated_output_path, 'videos')
				dated_alert_path = os.path.join(dated_output_path, 'alerts')

				# create output directory if doesn't exists already
				if not os.path.exists(dated_video_path):
					os.makedirs(dated_video_path)

				if not os.path.exists(dated_alert_path):
					os.makedirs(dated_alert_path)


				#############################################


				print('running frame',cap.get(1))
				images_comb=list(itertools.product(frame_list,[frame]))
				d_between.append(p.map(f,images_comb))
				Disp.clear_output(wait=True)
				norm_diff=np.log10(np.nanmedian(pd.DataFrame(d_between))/d_long_med)
				print(norm_diff)
				norm_diff_li.append(norm_diff)
				#plt.subplot(132), plt.plot(norm_diff_li)
				
				if len(d_between)>=size_short_pool: 
					d_between=d_between[1:]
					#--------------------fixing size of norm_diff_li_1 list as 500----------------------
					norm_diff_li=norm_diff_li[-500:]
					norm_diff_li_1=np.diff(norm_diff_li)
					norm_diff_1=norm_diff_li_1[-1]
					#thresh=runningMedian(norm_diff_li_1,int(size_short_pool*.5))[-2]
					# if ((eps is None ) or (len(norm_diff_li)<=eps_frame)):
					# 	eps=max(abs(norm_diff_li_1[1:]))

					if eps is None:
						if len(norm_diff_li)<=eps_frame:
							eps=max(abs(norm_diff_li_1))

					#thresh=runningMedian(norm_diff_li_1,int(size_short_pool))[-2]
					
					if not vandalised:
						norm_diff_li_1_good.append(norm_diff_li_1[-1])
						thresh_l=np.percentile(norm_diff_li_1_good[-thresh_shift_param*size_short_pool:],q=10)-thresh_shift_param*eps
						thresh_u=np.percentile(norm_diff_li_1_good[-thresh_shift_param*size_short_pool:],q=90)+thresh_shift_param*eps

					rest_down_period-=1
					# print(norm_diff_1,eps,thresh_l,thresh_u)
					#if ((norm_diff_1>max(.005,thresh*1.1)) & (rest_down_period<=0)) :
					if out_video is not None:
						out_video.write(frame)
					
					if ((norm_diff_1>thresh_u) and (rest_down_period<=0)) :
						#vandalised=True
						cv2.putText(frame,'vandalism Started',topLeft, font, fontScale,(0,0,255),lineType)                
						vandalised=True
						fp = open('alert.txt', 'a')
						fp.write('Event: vandalism started at: ' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + '\n')
						fp.close()

						if write_video:
							video_name = "camera_vandalism_"+str(dt.datetime.now().time().strftime('%H:%M:%S')) + ".avi"
							abs_video_path = os.path.join(dated_video_path, video_name)
							# out = cv2.VideoWriter(abs_video_path, fourcc, 15.0, (frame.shape[1], frame.shape[0]))
							# video_save_string = self.__path_to_output +"/"+ str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + "_" + self.__video_name + ".avi"
							out_video = cv2.VideoWriter(abs_video_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
							write_video = False

						# print("vandalism started")

					elif ((norm_diff_1<(thresh_l))& (vandalised)):
						
						fp = open('alert.txt', 'a')
						fp.write('Event: vandalism removed at: ' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + '\n')
						fp.close()

						print("vandalism removed")

						cv2.putText(frame,'vandalism removed',topLeft, font, fontScale,(0,255,255),lineType)
						vandalised=False
						rest_down_period=size_short_pool
					#elif norm_diff_1<(-.01):
						if out_video is not None:
							out_video.release()
							out_video = None
							write_video = True

					elif not vandalised:  
						print("no vandalism")
						cv2.putText(frame,'no vandalism',topLeft, font, fontScale,(0,255,0),lineType)
					else:

						fp = open('alert.txt', 'a')
						fp.write('Event: vandalism started but not removed' + '\n')
						fp.close()

						print("vandalism started but not removed")
						cv2.putText(frame,'vandalism started but not removed',topLeft, font, fontScale,(255,0,0),lineType)
	#                     u=np.percentile(norm_diff_li_1,q=80)
	#                     m=l=np.percentile(norm_diff_li_1,q=50)
	#                     l=np.percentile(norm_diff_li_1,q=20)
	#                     norm_diff_li_1[((norm_diff_li_1>l)&(norm_diff_li_1<u))]=m
					if plot:
						fig=plt.figure(figsize=(18, 4))
						plt.subplot(131), plt.imshow(frame.astype(np.uint8))
						plt.subplot(132),plt.plot(norm_diff_li)
						plt.subplot(133), plt.plot(norm_diff_li_1)
						plt.show()
				cv2.imshow("preview",frame)
				cv2.waitKey(1)

				#fig.savefig('frames_short_term/figure_'+str(frame_idx)+'.png')

				#print('normalized_diff=',norm_diff)
				frame_to_start+=skip_frames
				frame_idx+=1
				
				for i in range(skip_frames+1):
					ret,frame=cap.read()
					if resize_ratio>0:
						frame1=frame.copy()
						frame = cv2.resize(frame1, (0,0), fx=resize_ratio, fy=resize_ratio)

				
				# temporary testing condition
				if  (cv2.waitKey(1) & 0xFF == ord('q')):
					self.__quit_process = True
					break
			cap.release()
			cv2.destroyAllWindows()

		except Exception as e:
			print(e)
		p.close()
		cap.release()
		cv2.destroyAllWindows()


	def check_camera_vandalism(self):

		if self.__is_rtsp:

			urlInfo = self.__rtsp_url
			url = urlInfo.geturl()
			protocol = urlInfo.scheme
			host = urlInfo.hostname
			port = urlInfo.port
			location = urlInfo.netloc
			print url
			
			# req = "DESCRIBE rtsp://192.168.1.3:8554/output.mkv RTSP/1.0\r\nCSeq: 2\r\n\r\n"
			# req = "DESCRIBE rtsp://192.168.1.3:8554/ RTSP/1.0\r\nCSeq: 2\r\n\r\n"
			req = "DESCRIBE "+ url +" RTSP/1.0\r\nCSeq: 2\r\n\r\n"

			while True:
				
				try:
					s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
					s.connect((host, port))
					s.sendall(req)
					data = s.recv(100)
					# print data

					cap = cv2.VideoCapture(url)

					if not cap.isOpened():
						raise SystemError('Camera connected but video feed not available.')

					ret, frame = cap.read()
					if not ret:
						raise Exception('Camera connected but incoming images are invalid.')

					
					if not self.__quit_process:
						self.test_on_video('./videos/inside_office.avi',frame_to_start=0,file_name='inside_office_avi_video_org_',resize_ratio=0,plot=self.__plot,thresh_shift_param=3,skip_frames=0)
					else:	
						print("you manually terminated the process using quit value='q', please re-initialize the object.")

				except socket_error as serr:
					print "\nError-1: " + str(serr.args[1]) + ' at time: ' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
					time.sleep(2)
					fp = open('alert.txt', 'a')
					fp.write(str(serr.args[1]) + ' at time: ' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + '\n')
					fp.close()

				except SystemError as error:
					print "\nError-2: " + str(error) + ' at time: ' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
					time.sleep(2)
					fp = open('alert.txt', 'a')
					fp.write(str(error) + ' at time: ' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + '\n')
					fp.close()			

				except Exception as e:

					print "\nError-3: " + str(e) + ' at time: ' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
					time.sleep(2)
					fp = open('alert.txt', 'a')
					fp.write(str(e) + ' at time: ' + str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) + '\n')
					fp.close()
		
		elif self.__is_video:

			print("Running from a video file")
			print self.__file_name ,self.__video_name
			# sys.exit(0)
			self.test_on_video(self.__video_name, frame_to_start=self.__frame_to_start,file_name=self.__file_name, resize_ratio=self.__resize_ratio,plot=self.__plot,thresh_shift_param=self.__thresh_shift_param,skip_frames=self.__skip_frames)
			# self.test_on_video(self.__video_name, frame_to_start=0, file_name=self.__file_name, resize_ratio=self.__resize_ratio, plot=self.__plot,thresh_shift_param=self.__thresh_shift_param,skip_frames=self.__skip_frames)
		
		elif self.__is_camera:
			name = raw_input('Choose camera channel (0/1/2) ') or 0
			self.test_on_camera(channel=int(name), size_short_pool=self.__size_short_pool, f=diff_GDD_1,n_jobs=self.__n_jobs, plot=self.__plot, file_name=self.__file_name, thresh_shift_param=self.__thresh_shift_param,eps_frame=self.__eps_frame)