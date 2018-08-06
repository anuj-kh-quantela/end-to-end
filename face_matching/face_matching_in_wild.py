from sklearn import linear_model
import face_recognition
import cv2
import numpy as np
import os
import sys
import time
import datetime
import re


class FaceMatchingInWild(object):
	
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

		
		if not os.path.exists("unknown_faces"):
			os.makedirs("unknown_faces")

		
		self.__clf_sgd=linear_model.SGDClassifier(loss='log',n_jobs=7,shuffle=True,class_weight=None,
		warm_start=False,max_iter = np.ceil(10**6 / 600),average=True, tol=None)

		self.__load_unknown_faces = False
		
		print("Done Init!")
		


	def load_unknown_faces(self, u_faces_path='unknown_faces', number_of_times_to_upsample=2, num_jitters=2):
		
		start = time.time()
		
		self.__X_test = []
		self.__unknown_faces_list = os.listdir(u_faces_path)

		for img_name in self.__unknown_faces_list:
			img_path = os.path.join(u_faces_path, img_name)
			print("loading image: " + str(img_name))
			
			frame = cv2.imread(img_path)
			if max(frame.shape[0:2]) > 300.0: 
				scaling_factor = (300.0/max(frame.shape[0:2])) 
				frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

			converted_frame = frame[:, :, ::-1]
			
			face_locations = face_recognition.face_locations(converted_frame, number_of_times_to_upsample, model="cnn")
			face_encodings = face_recognition.face_encodings(converted_frame, face_locations, num_jitters)

			if len(face_encodings) > 0:
				self.__X_test.append(face_encodings[0])

		self.__load_unknown_faces = True



		end = time.time()
		print("time to load faces:" +str(end-start)) 

	# def __draw_rectangle(self, frame, face_locations, face_labels, scale=False):
	# 	for (top, right, bottom, left), name in zip(face_locations, face_labels):
	# 		# Scale back up face locations since the frame we detected in was scaled to 1/4 size
	# 		if scale:
	# 			top *= 2
	# 			right *= 2
	# 			bottom *= 2
	# 			left *= 2
	# 		# Draw a box around the face
	# 		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

	# 		# Draw a label with a name below the face
	# 		# cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
	# 		font = cv2.FONT_HERSHEY_DUPLEX
	# 		cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)



	def run_face_matching(self, increase_fps=None, number_of_times_to_upsample=1, num_jitters=1):

		
		if not self.__load_unknown_faces:
			print("Load unknown faces first before searching them!")
			sys.exit(0)
		
		
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

		cap = cv2.VideoCapture(self.__video_path)

		fps = int(cap.get(cv2.CAP_PROP_FPS))
		w = int(cap.get(3))
		h = int(cap.get(4))
		
		y_label_string = 'person_'
		count = 0

		m_countdown_time = 0
		m_writer_flag = None
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



			if inc_fps_flag:
				for i in range(num_frames_to_skip):
					cap.read()
	
			ret, frame = cap.read()
			if not ret:
				print("frames over..")
				break
			if max(frame.shape[0:2]) > 480.0: 
				scaling_factor = (480.0/max(frame.shape[0:2])) 
				frame = cv2.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

			rgb_small_frame = frame[:, :, ::-1]
			
			face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample, model="cnn")
			face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters)

			y_tr = [y_label_string+str(num) for num in range(0, len(face_encodings))]
			
			if len(y_tr) == 0:
				print("no face detected")
				m_writer_flag = None
				if out is not None:
					print("m_countdown_time: ", m_countdown_time)
					m_countdown_time -= 1
					if m_countdown_time == 0:
						out.release()
						out = None


			elif len(y_tr) == 1:
				
				try:

					match = face_recognition.compare_faces(self.__X_test, face_encodings[0])
					found_face_idx = np.where(match)[0]
					name = self.__unknown_faces_list[found_face_idx[0]]

					(top, right, bottom, left) = face_locations[0]
					cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

					# Draw a label with a name below the face
					# cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
					font = cv2.FONT_HERSHEY_DUPLEX
					cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)			

				except Exception as e:
					print e
					print "faces could not be detected"

			else:
				print( '\nFrame No: ' + str(count)  + " Found {} faces in the frame".format(len(y_tr)))
				
				try:
					self.__clf_sgd.fit(face_encodings, y_tr)
					y_pred = self.__clf_sgd.predict(self.__X_test)
					y_pred_proba = self.__clf_sgd.predict_proba(self.__X_test)
					max_prob = np.max(y_pred_proba, axis=1)
					max_prob_loc = np.argmax(y_pred_proba, axis=1)
					
					# IMPORTANT PART DO NOT DELETE
					face_indices = [(idx, face_loc_idx) for idx, (prob, face_loc_idx) in enumerate(zip(max_prob, max_prob_loc)) if prob > 0.9]
					for idx, loc_idx in face_indices:
						(top, right, bottom, left) = face_locations[loc_idx]
						cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
						name = self.__unknown_faces_list[idx]
					
						# Draw a label with a name below the face
						# cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
						font = cv2.FONT_HERSHEY_DUPLEX
						cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
						self.__write_video = True

					if ((m_writer_flag is None) and m_countdown_time == 0):
						m_writer_flag = 1
						video_name = "multiple_face_match_"+str(datetime.datetime.now().time().strftime('%H:%M:%S')) + ".avi"
						abs_video_path = os.path.join(dated_video_path, video_name)
						out = cv2.VideoWriter(abs_video_path, fourcc, 15.0, (frame.shape[1], frame.shape[0]))

					m_countdown_time = 60
					out.write(frame)

				except Exception as e:
					# if len(face_encodings) == 0:
					# 	# print len(face_encodings)
					# 	print('No face encoding found for the given face') 
					print e
			
			count += 1
			cv2.imshow("frame", frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
			
		if self.__write_video:
			# print("output video location: " + video_save_string)
			pass
		else:
			print("faces not found")

		cv2.destroyAllWindows()
		cap.release()


video_path = '/media/anuj/Work-HDD/WORK/CLOUD-DRIVE/Google-Drive/Computer-Vision/Sample-Videos/Face-Matching/inside_office_faces.avi'
vo = FaceMatchingInWild(video_path, 'bangalore', 'ecentric')
vo.load_unknown_faces()
vo.run_face_matching(increase_fps=(False, 0))