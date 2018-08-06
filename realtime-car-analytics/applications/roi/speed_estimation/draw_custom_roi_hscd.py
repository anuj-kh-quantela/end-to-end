import time
import sys
import os

import cv2
import configparser
from shapely.geometry import LineString

drawing_mode = False
select_x, select_y = -1, -1
boundary_points = []

distance = 0
frame_rate = 0
speed_limit = 0
DIVIDING_FACTOR = 3


# Function Name: parse_wkt_data
# Description: Parse wkt format string and extract (x,y) for inner polygon
# Input: (wkt_data: point in wkt format string)
# Output: Returns x,y point
def parse_wkt_data(wkt_data):
    """ Parse wkt format strng and extract (x,y) for inner polygon """
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
# Description: Draw rectangle using top-left and bottom-right corner points
# Input: (image: image on which line should be drawn, top-left point, bottom-right point, color: color of line in RGB,
#         thickness)
# Output: None. Draws rectangle using top-left and bottom-right corner points
def draw_rectangle(image, top_left, bottom_right, color, thickness):
    """ Draw line between two points p1 and p2 """
    cv2.rectangle(image, top_left, bottom_right, color, thickness)


# Function Name: save_config_file
# Description: Save all boundary points and parameters in config.ini file
# Input: (path_to_save: input video name)
# Output: Saves 'config.ini' file in data/<video_name>/config/ path
def save_config_file(path_to_save):
    """ Save all boundary points and parameters in config.ini file """
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read('./data/' + path_to_save + '/config/' + 'config.ini')

    config['HSCD'] = {}

    config['HSCD']['outer_bottom_left'] = str(boundary_points[0])
    config['HSCD']['outer_top_left'] = str(boundary_points[1])
    config['HSCD']['outer_top_right'] = str(boundary_points[2])
    config['HSCD']['outer_bottom_right'] = str(boundary_points[3])

    inner_bottom_left = set_boundary_points_inner(boundary_points[0], boundary_points[1])
    inner_top_left = set_boundary_points_inner(boundary_points[1], boundary_points[0])
    inner_top_right = set_boundary_points_inner(boundary_points[2], boundary_points[3])
    inner_bottom_right = set_boundary_points_inner(boundary_points[3], boundary_points[2])

    config['HSCD']['inner_bottom_left'] = str(inner_bottom_left)
    config['HSCD']['inner_top_left'] = str(inner_top_left)
    config['HSCD']['inner_top_right'] = str(inner_top_right)
    config['HSCD']['inner_bottom_right'] = str(inner_bottom_right)

    config['HSCD']['distance'] = str(distance)
    config['HSCD']['frame_rate'] = str(frame_rate)
    config['HSCD']['speed_limit'] = str(speed_limit)

    with open("./data/" + path_to_save + '/config/' + 'config.ini', 'w') as configfile:
        config.write(configfile)
    print('\n*** RoI configuration for High Speed Car Detection saved ***\n')


# Function Name: define_roi
# Description: Read mouse clicks and store RoI points. Take speed limit and frame rate as input and save in
#             'config.ini' file
# Input: (video_path: path to input video)
# Output: Saves RoI points, speed limit and frame rate in 'config.ini' file
def define_roi(video_path, pDistance, pFrameRate, pSpeedLimit):

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    """ Define roi (region of interest) for speed estimation """
    instruction_frame_path = 'applications/roi/speed_estimation/RoI_Instruction_Image_HSCD.jpg'
    roi_instruction_frame = cv2.imread(instruction_frame_path, -1)
    cv2.namedWindow('RoI Instructions', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('RoI Instructions', roi_instruction_frame)

    # wait to press Enter
    while True:
        if cv2.waitKey(10) & 0xFF == 13:
            break

    cv2.destroyWindow('RoI Instructions')
    cv2.waitKey(1)

    # open and read video file
    video = cv2.VideoCapture(video_path)
    if video.isOpened():
        rval, frame = video.read()
        copy_frame = frame.copy()
    else:
        print("*** Video error! Exiting... ***")
        sys.exit(1)

    global drawing_mode, distance, frame_rate, speed_limit

    def draw_roi(event, x, y, flags, param):
        """ Mouse callback function to draw roi """

        global boundary_points
        global drawing_mode, select_x, select_y

        if event == cv2.EVENT_LBUTTONDOWN:
            if drawing_mode:
                select_x, select_y = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            if drawing_mode:
                if len(boundary_points) < 4:
                    boundary_points.append((select_x, select_y))
                    if len(boundary_points) == 2:
                        draw_line(frame, boundary_points[0], boundary_points[1], (255, 255, 255), 2)
                    elif len(boundary_points) == 3:
                        draw_line(frame, boundary_points[1], boundary_points[2], (0, 255, 255), 2)
                    elif len(boundary_points) == 4:
                        draw_line(frame, boundary_points[2], boundary_points[3], (255, 255, 255), 2)
                        draw_line(frame, boundary_points[3], boundary_points[0], (0, 255, 255), 2)

    drawing_mode = True
    cv2.namedWindow('Define RoI', cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback('Define RoI', draw_roi, None)
    print("--------------------------------------------------------------")
    print("Define roi, select four points on a image")
    print("Please follow the below order while selecting roi")
    print("--------------------------------------------------------------")
    print("Select --> bottom left  point")
    print("Select --> top left point")
    print("Select --> top right point")
    print("Select --> bottom right point")
    print("---------------------------------------------------------------------")
    print("Press: \n'c': Confirm\n'r': Reset\n'q': Quit")

    while True:
        pos_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
        # display frame number
        cv2.putText(frame, "Frame No: {}".format(int(pos_frame)),
                    (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 255), 2)
        try:
            cv2.imshow('Define RoI', frame)
            key = cv2.waitKey(10) & 0xFF

            # for key "d", navigate forward
            if key == 100:
                rval, frame = video.read()
                copy_frame = frame.copy()
                boundary_points[:] = []

            # for key "a", navigate backward
            elif key == 97:
                video.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-2)
                rval, frame = video.read()
                copy_frame = frame.copy()
                boundary_points[:] = []

            # Reset if "r" is pressed
            elif key == ord("r"):
                frame = copy_frame.copy()
                boundary_points[:] = []

            # Confirm if "c" is pressed
            elif key == ord("c"):
                # Display error id RoI not set
                if len(boundary_points) < 4:
                    print("\n** ROI not set! Please set ROI to proceed. **")
                    print("or \nPress 'q' to Quit ROI setting")
                    frame = copy_frame.copy()
                    cv2.putText(frame, "ROI not set!",
                                (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(frame, "Scroll: 'a'or 'd'  Quit: 'q'",
                                (10, 90), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 255), 2)
                    boundary_points[:] = []

                else:
                    drawing_mode = False
                    path_to_save = "./data/" + video_name + '/config/'+ '/roi_sample_speed.jpg'
                    cv2.imwrite(path_to_save, frame)
                    cv2.destroyWindow('Define RoI')
                    cv2.waitKey(1)
                    print('\nsample image has been saved for future use as roi_sample.jpg')
                    break

            # Quit if "q" is pressed
            elif key == ord("q"):
                drawing_mode = False
                cv2.destroyWindow('Define RoI')
                cv2.waitKey(1)
                print('\n*** ROI NOT SET: Operation cancelled by user ***')
                sys.exit(0)

        except AttributeError:
            print("\n*** Reached end of video ***")
            print("Press 'a' to scroll back")
            video.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
            rval, frame = video.read()
            copy_frame = frame.copy()
            cv2.putText(frame, "Reached end of video!",
                        (10, 60), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, "Press 'a' to scroll back",
                        (10, 90), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0, 255, 255), 2)

        except cv2.error as e:
            print(e)
            sys.exit(1)

    time.sleep(1)

    distance = pDistance
    frame_rate = pFrameRate
    speed_limit = pSpeedLimit

    # while True:
    #     try:
    #         distance = float(input("\nPlease enter distance in meters: "))
    #     except ValueError:
    #         print("\nSorry, I didn't understand that! Try again.")
    #         continue
    #     if distance < 0 or distance == 0:
    #         print("\nSorry, distance must not be negative or zero!")
    #         continue
    #     else:
    #         break

    # while True:
    #     try:
    #         frame_rate = int(input("\nPlease enter frame rate of video in fps: "))
    #     except ValueError:
    #         print("\nSorry, I didn't understand that! Try again.")
    #         continue
    #     if frame_rate < 0 or frame_rate == 0:
    #         print("\nSorry, frame rate must not be negative or zero!")
    #         continue
    #     else:
    #         break

    # while True:
    #     try:
    #         speed_limit = int(input("\nPlease enter speed limit: "))
    #     except ValueError:
    #         print("\nSorry, I didn't understand that! Try again.")
    #         continue
    #     if speed_limit < 0 or speed_limit == 0:
    #         print("\nSorry, frame rate must not be negative or zero!")
    #         continue
    #     else:
    #         break

    save_config_file(video_name)
