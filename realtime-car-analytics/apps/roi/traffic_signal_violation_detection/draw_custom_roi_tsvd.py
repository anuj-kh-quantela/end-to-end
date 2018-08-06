import time
import sys
import os

import cv2
import configparser

drawing_mode = False
select_x, select_y = -1, -1
boundary_points = []


# Function Name: draw_line
# Description: Draw line between two points p1 and p2
# Input: (image: image on which line should be drawn, point p1, point p2, color: color of line in RGB, thickness)
# Output: None. Draws line between points p1 and p2
def draw_line(image, p1, p2, color, thickness):
    """ Draw line between two points p1 and p2 """
    cv2.line(image, p1, p2, color, thickness)


# Function Name: save_config_file
# Description: Save all boundary points and parameters in config.ini file under section
# Input: (path_to_save: input video name, where: traffic light/junction)
# Output: saves 'config.ini' file in the specified folder path
def save_config_file(path_to_save, where, boundary_points):
    """ Save all boundary points and parameters in config.ini file """

    # global boundary_points

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read('./data/' + path_to_save + '/config/' + 'config.ini')

    if where == 'light':
        try:
            config['RL'] = {}
            config['RL']['lower_left'] = str(boundary_points[0])
            config['RL']['upper_left'] = str(boundary_points[1])
            config['RL']['upper_right'] = str(boundary_points[2])
            config['RL']['lower_right'] = str(boundary_points[3])

            with open("./data/" + path_to_save + '/config/' + 'config.ini', 'w') as configfile:
                config.write(configfile)

            print('\n*** RoI Configuration for Red Light Signal saved ***\n')
        except IndexError:
            print("\n*** ERROR: Insufficient points for ROI. Redraw ROI! ***")
            sys.exit(0)

    elif where == 'zone':
        try:
            config['VZ'] = {}
            config['VZ']['lower_left'] = str(boundary_points[0])
            config['VZ']['upper_left'] = str(boundary_points[1])
            config['VZ']['upper_right'] = str(boundary_points[2])
            config['VZ']['lower_right'] = str(boundary_points[3])

            with open("./data/" + path_to_save + '/config/' + 'config.ini', 'w') as configfile:
                config.write(configfile)

            print('\n*** RoI Configuration for Red Light Violation Zone saved ***\n')
        except IndexError:
            print("\n*** ERROR: Insufficient points for ROI. Redraw ROI! ***")
            sys.exit(0)

    boundary_points[:] = []


# Function Name: define_roi
# Description: Read mouse clicks and store RoI points in 'config.ini' file
# Input: (video_path: path to input video, where: traffic light/junction)
# Output: saves RoI points in 'config.ini' file
def define_roi(video_path, where, user_boundary_points):
    # """ Define roi (region of interest) for speed estimation """

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # if where == 'light':
    #     instruction_frame_path = 'applications/roi/traffic_signal_violation_detection/RoI_Instruction_Image_RL.jpg'
    # else:
    #     instruction_frame_path = 'applications/roi/traffic_signal_violation_detection/RoI_Instruction_Image_VZ.jpg'

    # roi_instruction_frame = cv2.imread(instruction_frame_path, -1)
    # cv2.namedWindow('RoI Instructions', cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('RoI Instructions', roi_instruction_frame)

    # # wait to press Enter
    # while True:
    #     if cv2.waitKey(10) & 0xFF == 13:
    #         break

    # cv2.destroyWindow('RoI Instructions')
    # cv2.waitKey(1)

    # # open and read video file
    # video = cv2.VideoCapture(video_path)
    # if video.isOpened():
    #     rval, frame = video.read()
    #     copy_frame = frame.copy()
    # else:
    #     print("*** Video error! Exiting... ***")
    #     sys.exit(1)

    # global drawing_mode

    # def draw_rectangle(event, x, y, flags, param):
    #     """ Mouse callback function to draw roi """

    #     global boundary_points
    #     global drawing_mode, select_x, select_y

    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         if drawing_mode:
    #             select_x, select_y = x, y

    #     elif event == cv2.EVENT_LBUTTONUP:
    #         if drawing_mode:
    #             if len(boundary_points) < 4:
    #                 boundary_points.append((select_x, select_y))
    #                 if len(boundary_points) == 2:
    #                     draw_line(frame, boundary_points[0], boundary_points[1], (255, 255, 255), 2)
    #                 elif len(boundary_points) == 3:
    #                     draw_line(frame, boundary_points[1], boundary_points[2], (0, 255, 255), 2)
    #                 elif len(boundary_points) == 4:
    #                     draw_line(frame, boundary_points[2], boundary_points[3], (255, 255, 255), 2)
    #                     draw_line(frame, boundary_points[3], boundary_points[0], (0, 255, 255), 2)

    # drawing_mode = True
    # cv2.namedWindow('Define RoI', cv2.WINDOW_KEEPRATIO)
    # cv2.setMouseCallback('Define RoI', draw_rectangle)
    # print("Image resolution: ", frame.shape)
    # print("--------------------------------------------------------------")
    # print("Define roi, select four points on a image")
    # print("Please follow the below order while selecting roi")
    # print("--------------------------------------------------------------")
    # print("Select --> bottom left  point")
    # print("Select --> top left point")
    # print("Select --> top right point")
    # print("Select --> bottom right point")
    # print("---------------------------------------------------------------------")
    # print("Press: \n'c': Confirm\n'r': Reset\n'q': Quit")

    # while True:
    #     pos_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
    #     # display frame number
    #     cv2.putText(frame, "Frame No: {}".format(int(pos_frame)),
    #                 (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 255), 2)

    #     try:
    #         cv2.imshow('Define RoI', frame)
    #         key = cv2.waitKey(10) & 0xFF

    #         # for key "d", navigate forward
    #         if key == 100:
    #             rval, frame = video.read()
    #             copy_frame = frame.copy()
    #             boundary_points[:] = []

    #         # for key "a", navigate backward
    #         elif key == 97:
    #             video.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 2)
    #             rval, frame = video.read()
    #             copy_frame = frame.copy()
    #             boundary_points[:] = []

    #         if key == ord("r"):
    #             frame = copy_frame.copy()
    #             boundary_points[:] = []

    #         elif key == ord("c"):
    #             if len(boundary_points) < 4:
    #                 print("\n** ROI not set! Please set ROI to proceed. **")
    #                 print("or \nPress 'q' to Quit ROI setting")
    #                 frame = copy_frame.copy()
    #                 cv2.putText(frame, "ROI not set!",
    #                             (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 255), 2)
    #                 cv2.putText(frame, "Scroll: 'a'or 'd'  Quit: 'q'",
    #                             (10, 90), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 255), 2)
    #                 boundary_points[:] = []

    #             else:
    #                 drawing_mode = False

    #                 if where == 'light':
    #                     path_to_save = "./data/" + video_name + '/config/' + 'roi_sample_traffic_light.jpg'
    #                 else:
    #                     path_to_save = "./data/" + video_name + '/config/' + 'roi_sample_traffic_zone.jpg'

    #                 cv2.imwrite(path_to_save, frame)
    #                 cv2.destroyWindow('Define RoI')
    #                 cv2.waitKey(1)
    #                 if where == 'light':
    #                     print('\nSample image has been saved for future use as roi_sample_traffic_light.jpg')
    #                 else:
    #                     print('\nSample image has been saved for future use as roi_sample_traffic_zone.jpg')
    #                 break

    #         elif key == ord("q"):
    #             drawing_mode = False
    #             cv2.destroyWindow('Define RoI')
    #             cv2.waitKey(1)
    #             print('\n*** ROI NOT SET: Operation cancelled by user ***')
    #             sys.exit(0)

    #     except AttributeError:
    #         print("\n*** Reached end of video ***")
    #         print("Press 'a' to scroll back")
    #         video.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
    #         rval, frame = video.read()
    #         copy_frame = frame.copy()
    #         cv2.putText(frame, "Reached end of video!",
    #                     (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 255), 2)
    #         cv2.putText(frame, "Press 'a' to scroll back",
    #                     (10, 90), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 255), 2)

    #     except cv2.error as e:
    #         print(e)
    #         sys.exit(1)

    # time.sleep(1)
    save_config_file(video_name, where, user_boundary_points)
