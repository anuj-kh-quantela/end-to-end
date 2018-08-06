from camera_vandalism import CameraVandalism

obj = CameraVandalism('/home/anuj/git/work/video-analytics/video-analytics-api/city/bangalore/camera_vandalism/inside_office.avi', 'vijaywada', 'signal')
obj.test_on_video('/home/anuj/git/work/video-analytics/video-analytics-api/city/bangalore/camera_vandalism/inside_office.avi',n_jobs=1,plot=False)


