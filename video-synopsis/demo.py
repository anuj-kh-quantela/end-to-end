from video_synopsis import VideoSynopsis
from urllib.request import urlopen


# Video path, could be a local video as well as an RTSP feed.
path_to_video = './data/demoVideo.mp4' # Local video
# path_to_video = 'rtsp://192.168.1.104:8554/demoVideo.mpg' # RTSP link


video_object = VideoSynopsis(path_to_video, 'hyderabad', 'benz-circle')
video_object.get_video_length_util()



video_object.preprocess()

# # Time for which the video is to be synopsyzed
# time_split = "9 seconds"

# output_video_path = video_object.video_synopsis(time_split)
# print(output_video_path)

# # notebook_path = output_video_path.split('/')
# # print('https://'+urlopen('http://ip.42.pl/raw').read().decode("utf-8")+ ':8888/notebooks/'+'/'.join(notebook_path[notebook_path.index('video-analytics')+1:]))


# output_video_path = video_object.video_synopsis(user_def_split=2)
# print(output_video_path)


# # notebook_path = output_video_path.split('/')
# # print('https://'+urlopen('http://ip.42.pl/raw').read().decode("utf-8")+ ':8888/notebooks/'+'/'.join(notebook_path[notebook_path.index('video-analytics')+1:]))

# # Get list of unique objects in the video
# video_object.get_unique_objects()

# # Set query object name to be reverse searched in the video
# query_obj_name = "bus"

# # Reverse object search
# output_video_path = video_object.reverse_object_search(query_obj_name)
# print(output_video_path)


# # notebook_path = output_video_path.split('/')
# # print('https://'+urlopen('http://ip.42.pl/raw').read().decode("utf-8")+ ':8888/notebooks/'+'/'.join(notebook_path[notebook_path.index('video-analytics')+1:]))