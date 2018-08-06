import multiprocessing
from video_synopsis import VideoSynopsis

# @profile
def wrapper(video_path):
	vo = VideoSynopsis(video_path)
	vo.preprocess()
	# vo.get_video_length_util()
	# vo.video_synopsis("9 seconds")

# wrapper('./test-videos/obama.webm')


video_path = './data/traffic_video_two.avi'
if __name__ == '__main__':
	
	jobs = []
	for i in range(3):

		p = multiprocessing.Process(target=wrapper, args=(video_path,))
		jobs.append(p)
		p.start()