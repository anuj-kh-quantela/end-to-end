import cv2
import os
import sys
import subprocess
import skvideo.io
import skimage.io
import configparser


def yes_or_no(question):
    reply = (input(question+' (y/n): ')).lower().strip()
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        return yes_or_no("Uhhhhm... Please enter")


def split_video_frames(file_path, directory):
    try:
        video_data = skvideo.io.FFmpegReader(file_path)
        video_shape = video_data.getShape()
        print(video_shape)
        count = 0
        video_frames = video_data.nextFrame()
        for frame in video_frames:
            skimage.io.imsave('data/' + directory + '/frames/' + str(count) + '.jpg', frame)
            if count % 100 == 0:
                print('Read frame%d.jpg' % count)
            count += 1
        print("Video converted to frames")
        
        config = configparser.ConfigParser()
        config.optionxform = str
        config['Sequence'] = {}
        config['Sequence']['name'] = directory
        config['Sequence']['imDir'] = 'frames'
        config['Sequence']['frameRate'] = '20'
        config['Sequence']['seqLength'] = str(video_shape[0])
        config['Sequence']['imWidth'] = str(video_shape[2])
        config['Sequence']['imHeight'] = str(video_shape[1])
        config['Sequence']['imExt'] = '.jpg'
        
        path_to_save = './data/' + directory + '/config.ini'
        
        with open(path_to_save, 'w') as configfile:
            config.write(configfile)
        
    except RuntimeError:
        pass
    except ValueError:
        print('*** Error: Unable to read file ***')


def get_video_frames(data_file_path, override_existing=True):
    """
    Takes in a video MP4 file, given its filepath, and converts it to
    a series of ordered JPEG files, collected in a folder with a corresponding name.
    Returns the number of frames read in.
    If a frames directory already exists, just returns the number of files in that directory.
    Credit: http://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames
    """
    base = os.path.basename(data_file_path)
    directory_name = os.path.splitext(base)[0]
    path_directory_name = 'data/' + directory_name

    try:
        # if os.path.exists(path_directory_name):

        #     reply = yes_or_no('\nDirectory already exists. Do you want to overwrite?')
        #     if reply:
        #         command = 'rm -rf ' + path_directory_name + ' && ' + 'mkdir -p ' + path_directory_name + '/frames'
        #         subprocess.call(command, shell=True)
        #         split_video_frames(data_file_path, directory_name)

        #     else:
        #         print('\nOperation cancelled.')
        #         sys.exit(0)
        # else:
        #     command = 'mkdir -p ' + path_directory_name + '/frames'
        #     subprocess.call(command, shell=True)
        #     split_video_frames(data_file_path, directory_name)

        command = 'mkdir -p ' + path_directory_name + '/frames'
        subprocess.call(command, shell=True)
        split_video_frames(data_file_path, directory_name)

    except KeyboardInterrupt:
        print('\n*** Error occured! Please try again.***')
        sys.exit(1)


get_video_frames(sys.argv[1])
