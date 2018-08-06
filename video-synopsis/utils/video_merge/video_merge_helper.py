import numpy as np
import cv2
import glob, os, shutil, sys

def video_merge(video_path, numSplits=2, start_time=""):

	video_name = os.path.splitext(os.path.basename(video_path))[0]
	cap = cv2.VideoCapture(video_path)

	if not cap.isOpened():
		print('could not open video stream' , video_path)
		exit(1)

	numFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
	frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

	
	# code to delete all the existing directories beginning with dir_*
	for path in glob.glob("dir_*"):
	    shutil.rmtree(path)

	directory = []
	for numDir in range(0, numSplits):
		directory.append("dir_"+str(numDir))
		if not os.path.exists(directory[numDir]):
			os.makedirs(directory[numDir])

	numFramesPerDir = int(numFrames/numSplits)

	count = 0
	dirCount = 0
	for numFrames in range(0, numFramesPerDir*numSplits):
		ret, frame = cap.read()
		if not count == numFramesPerDir:
			pass
		else:
			count = 0
			dirCount = dirCount + 1
		cv2.imwrite(str(str(directory[dirCount]) + "/" + 'frame%d.jpg' % count), frame)
		# print( "Processing " + str(str(directory[dirCount]) + "/" + 'frame%d.jpg' % count) )
		print( "Processing " + ('frame%d.jpg' % count) + " for split number: " + str(dirCount) )
		count = count + 1
	cap.release()

	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	out = cv2.VideoWriter(start_time+"_merged-"+video_name+".avi", fourcc, 15.0, (frameWidth, frameHeight))
	print("\n ------------------------------------ \n")
	for frameNum in range(0, numFramesPerDir):
		wFrame = np.zeros((frameHeight,frameWidth,3), np.uint8)
		for dirNum in range(0, numSplits):
			img = cv2.imread(str(str(directory[dirNum]) + "/" + 'frame%d.jpg' % frameNum))
			# numpy behaves differently when an array is divided and when it is multiplied by a scalar
			# for some reason, scalar multiplication gives a completely black image 
			# therefore using division
			imgMod = img.copy()//numSplits
			print( "Merging frames " + str(str(directory[dirNum]) + "/" + 'frame%d.jpg' % frameNum))
			wFrame = wFrame + imgMod
		out.write(wFrame)
		# cv2.imshow("merged frames", wFrame)
		# cv2.waitKey(1)

	out.release()
	
video_merge(sys.argv[1], int(sys.argv[2]), sys.argv[3])