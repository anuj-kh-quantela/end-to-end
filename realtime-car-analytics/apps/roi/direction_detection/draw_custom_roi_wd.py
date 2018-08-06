import time
import sys
import os

import cv2
import numpy as np
import configparser
from shapely.geometry import LineString

drawing_mode = False
select_x, select_y = -1, -1
boundary_points_first = []
boundary_points_second = []
frame_roi = np.array([])
DIVIDING_FACTOR = 4
arrow_head_xy_first = [0, 0]
arrow_tail_xy_first = [0, 0]
arrow_head_xy_second = [0, 0]
arrow_tail_xy_second = [0, 0]
road = 0


# Function Name: parse_wkt_data
# Description: Parse wkt format string and extract (x,y) for inner polygon
# Input: (wkt_data: point in wkt format string)
# Output: Returns x,y point
def parse_wkt_data(wkt_data):
    """ Parse wkt format string and extract (x,y) for inner polygon """
    wkt_data_split = wkt_data.split()
    x = float(wkt_data_split[1].strip('('))
    y = float(wkt_data_split[2].strip(')'))
    x = int(round(x))
    y = int(round(y))
    return x, y


# Function Name: set_boundary_points_inner
# Description: Find (x,y) point on a line joining the points pa and pb at a distance y_threshold
# Input: line points pa and pb
# Output: Returns x and y points of the point on the line joining the points pa and pb at a distance y_threshold
def set_boundary_points_inner(pa, pb):
    """ Find (x,y) point on a line joining the points pa and pb at a distance y_threshold """
    line_distance = LineString([pa, pb])
    y_threshold = int(line_distance.length / DIVIDING_FACTOR)
    point_y_threshold = LineString([pa, pb]).interpolate(y_threshold)
    (point_x, point_y) = parse_wkt_data(point_y_threshold.wkt)
    return point_x, point_y


# Function Name: draw_line
# Description: Draw line between two points p1 and p2
# Input: (image: image on which line should be drawn, point p1, point p2, color: color of line in RGB, thickness)
# Output: None. Draws line between points p1 and p2
def draw_line(image, p1, p2, color, thickness):
    """ Draw line between two points p1 and p2 """
    cv2.line(image, p1, p2, color, thickness)


# Function Name: draw_rectangle
# Description: Draw rectangle using top left and bottom right points
# Input: (image: image on which line should be drawn, point top_left, point bottom_right,
#        color: color of line in RGB, thickness)
# Output: None. Draws rectangle using top left and bottom right points
def draw_rectangle(image, top_left, bottom_right, color, thickness):
    """ Draw rectangle using top left and bottom right points """
    cv2.rectangle(image, top_left, bottom_right, color, thickness)


# Function Name: draw_arrow_point
# Description: Draw arrow between two points p and q
# Input: (image: image on which line should be drawn, point p, point q, color: color of line in RGB, thickness)
# Output: None. Draws arrow between points p and q
def draw_arrow_point(frame, p, q, color, thickness):
    cv2.arrowedLine(frame, p, q, color, thickness)


# Function Name: draw_arrow
# Description: Draw arrow between two points p and q
# Input: (image: image on which line should be drawn, point p, point q, color: color of line in RGB,
#         arrow_magnitude, thickness, line_type, shift)
# Output: None. Draws arrow between points p and q
def draw_arrow(image, p, q, color, arrow_magnitude=50, thickness=2, line_type=8, shift=0):
    # adapted from http://mlikihazar.blogspot.com.au/2013/02/draw-arrow-opencv.html

    # draw arrow tail
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # calc angle of the arrow
    angle = np.arctan2(p[1]-q[1], p[0]-q[0])
    # starting point of first line of arrow head
    p = (int(q[0] + arrow_magnitude * np.cos(angle + np.pi/4)),
         int(q[1] + arrow_magnitude * np.sin(angle + np.pi/4)))
    # draw first half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # starting point of second line of arrow head
    p = (int(q[0] + arrow_magnitude * np.cos(angle - np.pi/4)),
         int(q[1] + arrow_magnitude * np.sin(angle - np.pi/4)))
    # draw second half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)


# Function Name: find_mid_point
# Description: Calculate mid-point on line joining points pa and pb
# Input: (pa, pb)
# Output: Returns coordinates of mid-point
def find_mid_point(pa, pb):
    line_distance = LineString([pa, pb])
    mid_distance = line_distance.length * 0.5
    mid_point = LineString([pa, pb]).interpolate(mid_distance)
    (point_x, point_y) = parse_wkt_data(mid_point.wkt)
    return point_x, point_y


# Function Name: save_config_file
# Description: Save all boundary points and parameters in config.ini file
# Input: (path_to_save: input video name)
# Output: Saves 'config.ini' file in data/<video_name>/config/ path
# save_config_file(video_name, boundary_points_first, boundary_points_second, road)
def save_config_file(path_to_save, boundary_points_first, boundary_points_second, arrow_tail_xy_first, arrow_head_xy_first, arrow_tail_xy_second, arrow_head_xy_second, road):
# def save_config_file(path_to_save, boundary_points_first, boundary_points_second, arrow_tail_xy_first, arrow_head_xy_first, arrow_tail_xy_second, arrow_head_xy_second, road):
    """ Save all boundary points and parameters in config.ini file """
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read('./data/' + path_to_save + '/config/' + 'config.ini')

    try:
        config['DD'] = {}
        config['DD']['lower_left_first'] = str(boundary_points_first[0])
        config['DD']['upper_left_first'] = str(boundary_points_first[1])
        config['DD']['upper_right_first'] = str(boundary_points_first[2])
        config['DD']['lower_right_first'] = str(boundary_points_first[3])

        config['DD']['arrow_head_xy_first'] = str(arrow_head_xy_first)
        config['DD']['arrow_tail_xy_first'] = str(arrow_tail_xy_first)

        if road == 2:
            config['DD']['lower_left_second'] = str(boundary_points_second[0])
            config['DD']['upper_left_second'] = str(boundary_points_second[1])
            config['DD']['upper_right_second'] = str(boundary_points_second[2])
            config['DD']['lower_right_second'] = str(boundary_points_second[3])
            config['DD']['arrow_head_xy_second'] = str(arrow_head_xy_second)
            config['DD']['arrow_tail_xy_second'] = str(arrow_tail_xy_second)

        config['DD']['road'] = str(road)

        with open("./data/" + path_to_save + '/config/' + 'config.ini', 'w') as configfile:
            config.write(configfile)
        print('\n*** RoI Configuration for Wrong Direction Detection saved ***\n')
    except IndexError:
        print("\n*** ERROR: Insufficient points for ROI. Redraw ROI! ***")
        sys.exit(0)


# Function Name: define_roi
# Description: Read mouse clicks and store RoI points in 'config.ini' file
# Input: (video_path: Path to input video)
# Output: saves RoI points in 'config.ini' file
def define_roi(video_path, user_boundary_points):
    """ Define roi (region of interest) for speed estimation """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # instruction_frame_path = 'applications/roi/direction_detection/RoI_Instruction_Image_WD.jpg'
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

    # global drawing_mode, road
    global road
    global boundary_points_first, boundary_points_second
    global arrow_head_xy_first, arrow_tail_xy_first, arrow_head_xy_second, arrow_tail_xy_second
    # def draw_roi(event, x, y, flags, param):
    #     """ Mouse callback function to draw roi """

        # global boundary_points_first, boundary_points_second
        # global arrow_head_xy_first, arrow_tail_xy_first, arrow_head_xy_second, arrow_tail_xy_second
    #     global drawing_mode, select_x, select_y, frame_roi

    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         if drawing_mode:
    #             select_x, select_y = x, y

    #     elif event == cv2.EVENT_LBUTTONUP:
    #         if drawing_mode:
    #             if len(boundary_points_first) < 4:
    #                 boundary_points_first.append((select_x, select_y))
    #                 if len(boundary_points_first) == 2:
    #                     draw_line(frame, boundary_points_first[0], boundary_points_first[1], (255, 255, 255), 2)
    #                 elif len(boundary_points_first) == 3:
    #                     draw_line(frame, boundary_points_first[1], boundary_points_first[2], (0, 255, 255), 2)
    #                 elif len(boundary_points_first) == 4:
    #                     draw_line(frame, boundary_points_first[2], boundary_points_first[3], (255, 255, 255), 2)
    #                     draw_line(frame, boundary_points_first[3], boundary_points_first[0], (0, 255, 255), 2)
    #                     arrow_tail_xy_first = find_mid_point(boundary_points_first[0], boundary_points_first[3])
    #                     arrow_head_xy_first = find_mid_point(boundary_points_first[1], boundary_points_first[2])
    #                     draw_arrow(frame, arrow_tail_xy_first, arrow_head_xy_first, (255, 255, 255))
    #                     # global frame_roi
    #                     frame_roi = frame.copy()

    #             elif len(boundary_points_first) == 4 and len(boundary_points_second) < 4:
    #                 boundary_points_second.append((select_x, select_y))
    #                 if len(boundary_points_second) == 2:
    #                     draw_line(frame, boundary_points_second[0], boundary_points_second[1], (255, 255, 255), 2)
    #                 elif len(boundary_points_second) == 3:
    #                     draw_line(frame, boundary_points_second[1], boundary_points_second[2], (0, 255, 255), 2)
    #                 elif len(boundary_points_second) == 4:
    #                     draw_line(frame, boundary_points_second[2], boundary_points_second[3], (255, 255, 255), 2)
    #                     draw_line(frame, boundary_points_second[3], boundary_points_second[0], (0, 255, 255), 2)
    #                     arrow_tail_xy_second = find_mid_point(boundary_points_second[0], boundary_points_second[3])
    #                     arrow_head_xy_second = find_mid_point(boundary_points_second[1], boundary_points_second[2])
    #                     draw_arrow(frame, arrow_tail_xy_second, arrow_head_xy_second, (255, 255, 255))

    # drawing_mode = True
    # cv2.namedWindow('Define RoI', cv2.WINDOW_KEEPRATIO)
    # cv2.setMouseCallback('Define RoI', draw_roi)
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
    #             boundary_points_first[:] = []
    #             boundary_points_second[:] = []

    #         # for key "a", navigate backward
    #         elif key == 97:
    #             video.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 2)
    #             rval, frame = video.read()
    #             copy_frame = frame.copy()
    #             boundary_points_first[:] = []
    #             boundary_points_second[:] = []

    #         if key == ord("r"):
    #             frame = copy_frame.copy()
    #             boundary_points_first[:] = []
    #             boundary_points_second[:] = []

    #         elif key == ord("c"):
    #             if len(boundary_points_first) < 4:
    #                 print("\n** ROI not set! Please set ROI to proceed. **")
    #                 print("or \nPress 'q' to Quit ROI setting")
    #                 frame = copy_frame.copy()
    #                 cv2.putText(frame, "ROI not set!",
    #                             (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 255), 2)
    #                 cv2.putText(frame, "Scroll: 'a'or 'd'  Quit: 'q'",
    #                             (10, 90), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 255), 2)
    #                 boundary_points_first[:] = []

    #             else:
    #                 drawing_mode = False

    #                 path_to_save = "./data/" + video_name + '/config/'+ 'roi_sample_direction.jpg'
    #                 if len(boundary_points_second) < 4:
    #                     road = 1
    #                     boundary_points_second[:] = []
    #                     frame = frame_roi.copy()
    #                 else:
    #                     road = 2
    #                 cv2.imwrite(path_to_save, frame)
    #                 cv2.destroyWindow('Define RoI')
    #                 cv2.waitKey(1)
    #                 print('\n*** RoI set for {} road(s) ***'.format(road))
    #                 print('\nSample image has been saved for future use as roi_sample_direction.jpg')
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
    if len(user_boundary_points) == 4:
        for pts in user_boundary_points:
            boundary_points_first.append(pts)

        arrow_tail_xy_first = find_mid_point(boundary_points_first[0], boundary_points_first[3])
        arrow_head_xy_first = find_mid_point(boundary_points_first[1], boundary_points_first[2])
         
        road = 1
    elif len(user_boundary_points) == 8:
        for pts in user_boundary_points[:4]:
            boundary_points_first.append(pts)

        arrow_tail_xy_first = find_mid_point(boundary_points_first[0], boundary_points_first[3])
        arrow_head_xy_first = find_mid_point(boundary_points_first[1], boundary_points_first[2])


        for pts in user_boundary_points[4:]:
            boundary_points_second.append(pts)


        arrow_tail_xy_second = find_mid_point(boundary_points_second[0], boundary_points_second[3])
        arrow_head_xy_second = find_mid_point(boundary_points_second[1], boundary_points_second[2])

        road = 2

    save_config_file(video_name, boundary_points_first, boundary_points_second, arrow_tail_xy_first, arrow_head_xy_first, arrow_tail_xy_second, arrow_head_xy_second, road)