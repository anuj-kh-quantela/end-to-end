import time
import sys
import cv2
from shapely.geometry import LineString

drawing_mode = False
select_x, select_y = -1, -1
boundary_points = []
boundary_points_inner = []
distance = 0
frame_rate = 0
DIVIDING_FACTOR = 4

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



# Function Name: define_roi
# Description: Read mouse clicks and store RoI points. Take speed limit and frame rate as input and save in 'config_roi.ini' file
# Input: (frame: image on which RoI should be drawn, copy_frame: copy of the frame, folder_path)
# Output: saves RoI points, speed limit and frame rate in 'config_roi.ini' file
def define_roi(frame, copy_frame):
    """ Define roi (region of interest) for speed estimation """

    global drawing_mode, distance, frame_rate

    def draw_rectangle(event, x, y, flags, param):
        """ Mouse callback function to draw roi """

        global boundary_points, boundary_points_inner
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

                        print("---------------------------------------------------------------------")
                        # (x1_bottom, y1_bottom) = set_boundary_points_inner(boundary_points[0], boundary_points[1])
                        # boundary_points_inner.append((x1_bottom, y1_bottom))
                        # (x1_up, y1_up) = set_boundary_points_inner(boundary_points[1], boundary_points[0])
                        # boundary_points_inner.append((x1_up, y1_up))
                        # (x2_up, y2_up) = set_boundary_points_inner(boundary_points[2], boundary_points[3])
                        # boundary_points_inner.append((x2_up, y2_up))
                        # (x2_bottom, y2_bottom) = set_boundary_points_inner(boundary_points[3], boundary_points[2])
                        # boundary_points_inner.append((x2_bottom, y2_bottom))


                        # draw_line(frame, boundary_points_inner[0], boundary_points_inner[3], (255, 255, 204), 2)
                        # draw_line(frame, boundary_points_inner[1], boundary_points_inner[2], (255, 255, 204), 2)
                        # print("\ninner boundary", boundary_points_inner)
                        # print("---------------------------------------------------------------------")
                        print("Press: \n'c': Confirm\n'r': Reset\n'q': Quit")
                        # print("\n outer boundary", boundary_points)
                        print("--------------------------------------------------------------------- \n")


    drawing_mode = True
    cv2.namedWindow('Define RoI', cv2.WINDOW_KEEPRATIO)
    cv2.setMouseCallback('Define RoI', draw_rectangle)
    print("\nImage resolution: ", frame.shape)
    print("--------------------------------------------------------------")
    print("Define roi, select Left->Top->Right->Bottom points on an image")
    print("Please follow the below order while selecting roi")
    print("--------------------------------------------------------------")
    print("Select --> lower_left_x, lower_left_y  point")
    print("Select --> upper_left_x, upper_left_y  point")
    print("Select --> upper_right_x, upper_right_y  point")
    print("Select --> lower_right_x, lower_right_y  point")

    while True:
        cv2.imshow('Define RoI', frame)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("r"):
            frame = copy_frame.copy()
            boundary_points[:] = []
            boundary_points_inner[:] = []
        elif key == ord("c"):
            drawing_mode = False
            # path_to_save = './data/' + folder_path + '/roi_sample.jpg'
            # cv2.imwrite(path_to_save, frame)
            # for i in range(0, 4):
            cv2.destroyWindow('Define RoI')
            cv2.waitKey(1)
            # print('\nsample image has been saved for future use as roi_sample.jpg')
            break
        elif key == ord("q"):
            drawing_mode = False
            # for i in range(0, 4):
            cv2.destroyWindow('Define RoI')
            cv2.waitKey(1)
            print('\n*** ROI NOT SET: Operation cancelled by user ***')
            sys.exit(0)

    time.sleep(1)

    return boundary_points
    # save_config_file(folder_path)

