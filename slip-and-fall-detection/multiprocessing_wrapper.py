import multiprocessing
from slip_and_fall import SlipAndFall


# city_arr = ['city1', 'city2', 'city3', 'city4', 'city5']
# location_arr = ['loc1', 'loc2', 'loc3', 'loc4']

city_str = 'city'
loc_str = 'loc'


def wrapper(video_path, city, location):
	vo = SlipAndFall(video_path, city, location)
	vo.slip_and_fall_algo(plot_intermediate=True,remove_shadow=True, minm_area=8370, maxm_area=90360, verbose=True)


video_path = 'test-4.webm'

if __name__ == '__main__':
	
	jobs = []
	for i in range(10):

		p = multiprocessing.Process(target=wrapper, args=(video_path, city_str+str(i), loc_str+str(i)))
		jobs.append(p)
		p.start()
	