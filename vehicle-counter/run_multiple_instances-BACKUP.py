import multiprocessing
from traffic_density_working_api import VehicleCounter
import numpy as np


def wrapper(video_channel, city_name, location_name, minm_area, roi):
		
	vo = VehicleCounter(video_channel, city_name, location_name)
	vo.num_vehicle(video_channel, minm_area=minm_area,roi=roi,plot_intermediate=False)


video_channel_arr = ['rtsp://192.168.20.9/78c6c91c-e9a3-4fa0-ab67-02942691aede/78c6c91c-e9a3-4fa0-ab67-02942691aede_vs1?token=78c6c91c-e9a3-4fa0-ab67-02942691aede^LVERAMOTD^50^26^26^1657790789^37cc0b67591f2083b984e614cf8b49685695c37a&username=admin', 'rtsp://192.168.20.10/956fce7e-06ea-457a-bca3-a376ec6c653a/956fce7e-06ea-457a-bca3-a376ec6c653a_vs1?token=956fce7e-06ea-457a-bca3-a376ec6c653a^LVERAMOTD^50^26^26^1657790789^ad8d6de28376a2cf1f87acaf5a124737ac8261d3&username=admin', 'rtsp://192.168.20.10/3e2145e8-1bdd-488d-84e0-804b77856cb6/3e2145e8-1bdd-488d-84e0-804b77856cb6_vs1?token=3e2145e8-1bdd-488d-84e0-804b77856cb6^LVERAMOTD^50^26^26^1657790795^5ff35815ba861a1be2a645e4b906b904384dc162&username=admin', 'rtsp://192.168.20.10/329eb34a-55a8-487b-892a-03b156ecb99f/329eb34a-55a8-487b-892a-03b156ecb99f_vs1?token=329eb34a-55a8-487b-892a-03b156ecb99f^LVERAMOTD^50^26^26^1657790796^5700e4b1620958da183f94174731f59b96b84066&username=admin', 'rtsp://192.168.20.9/a05e21a0-8d74-4cdc-90fc-359e25228d93/a05e21a0-8d74-4cdc-90fc-359e25228d93_vs1?token=a05e21a0-8d74-4cdc-90fc-359e25228d93^LVERAMOTD^50^26^26^1657790797^0c82fb4ad8f6ce716409f62e805449a30dba1246&username=admin']
location_name_arr = ['akashwani_east', 'Benzcircle-1', 'Deputy-01', 'DV-Manor-2', 'Fortune-Murali-Right']
minm_area_arr = [6825, 42008, 28851, 23718, 13000]
roi_arr = [[(620, 150), (892, 228), (681, 623), (27, 471)], [(367, 302), (1157, 304), (1895, 834), (419, 1030)], [(620, 552), (1174, 563), (1142, 1022), (65, 777)], [(921, 155), (1325, 162), (1887, 498), (932, 543)], [(546, 271), (1092, 304), (1635, 690), (543, 729)]]


video_channel_test_arr = ['akashwani.mp4', 'Benzcircle-1.avi', 'Deputy-01.avi', 'DV Manor-2.avi', 'Fortune-Murali-Right.avi']
location_name_test_arr = ['akashwani_east', 'Benzcircle-1', 'Deputy-01', 'DV-Manor-2', 'Fortune-Murali-Right']
minm_area_test_arr = [6825, 42008, 28851, 23718, 13000]
roi_test_arr = [[(620, 150), (892, 228), (681, 623), (27, 471)], [(367, 302), (1157, 304), (1895, 834), (419, 1030)], [(620, 552), (1174, 563), (1142, 1022), (65, 777)], [(921, 155), (1325, 162), (1887, 498), (932, 543)], [(546, 271), (1092, 304), (1635, 690), (543, 729)]]

if __name__ == '__main__':
	
	jobs = []
	for i in range(len(minm_area_arr)):
		video_channel = video_channel_arr[i]
		location_name = location_name_arr[i]
		minm_area = minm_area_arr[i]
		roi = np.array(roi_arr[i], dtype = np.float32)

		p = multiprocessing.Process(target=wrapper, args=(video_channel, 'hyderabad', location_name, minm_area, roi,))
		jobs.append(p)
		p.start()

	# jobs = []
	# for i in range(len(minm_area_test_arr)):
	# 	video_channel = video_channel_test_arr[i]
	# 	location_name = location_name_test_arr[i]
	# 	minm_area = minm_area_test_arr[i]
	# 	roi = np.array(roi_test_arr[i], dtype = np.float32)

	# 	p = multiprocessing.Process(target=wrapper, args=(video_channel, 'hyderabad', location_name, minm_area, roi,))
	# 	jobs.append(p)
	# 	p.start()


