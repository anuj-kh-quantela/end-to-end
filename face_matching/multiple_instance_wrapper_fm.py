import multiprocessing
from face_matching_in_wild import FaceMatchingInWild


def wrapper(video_path):
	vo = FaceMatchingInWild(video_path)
	vo.load_unknown_faces()
	vo.run_face_matching()


video_path = './data/outside_office_faces.avi'
if __name__ == '__main__':
	
	jobs = []
	for i in range(1):
		p = multiprocessing.Process(target=wrapper, args=(video_path,))
		jobs.append(p)
		p.start()