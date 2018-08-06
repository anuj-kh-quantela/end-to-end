import multiprocessing
from camera_vandalism import CameraVandalism
import multiprocessing
import time


def wrapper():
	print "came into the wrapper"
	# video_object = CarAnalytics(video_path)
	# video_object.run()
	obj = CameraVandalism('/media/anuj/Work-HDD/WORK/CLOUD-DRIVE/Google-Drive/Computer-Vision/Sample-Videos/Camera-Vandalism/inside_office.avi')
	obj.test_on_video('/media/anuj/Work-HDD/WORK/CLOUD-DRIVE/Google-Drive/Computer-Vision/Sample-Videos/Camera-Vandalism/inside_office.avi',n_jobs=1,plot=False)
	

	


jobs = []

for i in range(1):
	p = multiprocessing.Process(target=wrapper, args=())
	jobs.append(p)
	p.start()

print "I AM DONE "