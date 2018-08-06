"""
Runs yolo on multiple images from a directory.

IN: dir containing multiple images
OUT: standard yolo output

Run instruction:
python run-yolo.py <dir>
"""

import sys
import os
import subprocess
import re

cwd = os.getcwd()
os.chdir(cwd)
#subprocess.call("mkdir output", shell=True)

if len(sys.argv) < 2:
	print "Run instruction: python runtest.py <dir>\nABORTING"
	sys.exit()

run_yolo_command = "./darknet detect cfg/yolo.cfg yolo.weights "
#rename_file_command = "mv *.png output/"

# Directory of images
dir_in = sys.argv[1]

# Loop over each image file in given directory

length_images = len(os.listdir(dir_in))

f = open('framesize.txt', 'w')
f.write(str(length_images))  
f.close()

subprocess.call(run_yolo_command + " " + dir_in, shell=True)
#subprocess.call(rename_file_command + str(file)[:-4] + ".png", shell=True)
