import numpy as np
import cv2
import configparser
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# global
light_coordinates = []
zone_coordinates = []
cars_in_not_red = np.array([])
dir_count = {}  # car_id: dir_result_count
read_config_file_points = False


# Function Name: is_right_dir
# Description: determines the direction of object id, returns True if direction is away from camera
# Input: direction flag
# Output: True or False
def is_right_dir(direction):
    if direction == 0:
        return True
    else:
        return False


# Function Name: get_centroid
# Description: takes bounding box coordinates (x1,y1,x2,y2) [top left, bottom right points] and returns centroid of the same
# Input: bounding box coordinates
# Output: centroid point
def get_centroid(bounding_box):
    # find centroid
    reshape_bbox = np.reshape(bounding_box, (2, 2))
    reshape_bbox[1] = (reshape_bbox[1] - reshape_bbox[0]) / 2
    centroid = np.sum(reshape_bbox, axis=0)
    return centroid


# Function Name: is_in_violation_zone
# Description: reads config file and checks if centroid of bounding box is present in Zone ROI
# Input: bounding box, video name
# Output: True/False
def is_in_violation_zone(bbox, video_name):
    global zone_coordinates
    centroid = get_centroid(bbox)

    if len(zone_coordinates) == 0:
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read("./data/" + video_name + "/config/" + "/config.ini")
        coordinates = config['VZ']

        lower_left = eval(coordinates['lower_left'])
        upper_left = eval(coordinates['upper_left'])
        upper_right = eval(coordinates['upper_right'])
        lower_right = eval(coordinates['lower_right'])

        zone_coordinates.append(lower_left)
        zone_coordinates.append(upper_left)
        zone_coordinates.append(upper_right)
        zone_coordinates.append(lower_right)

    polygon = Polygon(zone_coordinates)
    if polygon.contains(Point(centroid)):
        return True
    else:
        return False



# Function Name: detect_color
# Description: takes cv2 image as input and determines if red is present in the image or not
# Input: cv2 image
# Output: True/False
def detect_color(cropped_image):
    # convert image to hsv
    image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)

    # hsv boundaries for colors
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # apply red mask for red detection
    # inRange converts all pixels in image falling between lower_red and upper_red to white pixels
    red_mask = cv2.inRange(image, lower_red, upper_red)

    # check presence of white pixels
    if np.count_nonzero(red_mask):
        return True

    else:
        return False


# Function Name: crop
# Description: given an image and boundary points as x,y,w,h, this module extracts pixels in the bounding box
# Input: image file and boundary points
# Output: cropped image based on boundary points 
def crop(frame):
    global light_coordinates

    # print "light_coordinates['upper_left']", type(eval(light_coordinates['upper_left'])[0])

    ulx = eval(light_coordinates['upper_left'])[0]
    uly = eval(light_coordinates['upper_left'])[1]
    lrx = eval(light_coordinates['lower_right'])[0]
    lry = eval(light_coordinates['lower_right'])[1]
    llx = eval(light_coordinates['lower_left'])[0]
    lly = eval(light_coordinates['lower_left'])[1]
    urx = eval(light_coordinates['upper_right'])[0]
    ury = eval(light_coordinates['upper_right'])[1]

    x = ulx
    y = uly
    w = urx - ulx
    h = lly - uly

    cropped_frame = frame[y:(y + h), x:(x + w)]

    return cropped_frame


# Function Name: is_red
# Description: crops frame, determines if color red is present in the cropped frame or not
# Input: cv2 image, video name
# Output: True/False
def is_red(frame, video_name):
    global light_coordinates

    if len(light_coordinates) == 0:
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read("./data/" + video_name + "/config/" + "/config.ini")
        light_coordinates = config['RL']

    cropped_frame = crop(frame)

    return detect_color(cropped_frame)


# Function Name: detect_red_violation
# Description: detects if given car point violates traffic signal or not
# Input: video_file_name, frame, frame_id, car_id, bbox, direction
# Output:
#   return 0 : no violation
#   return 1 : violation
#   return -1: can't say
def detect_red_violation(video_file_name, frame, frame_id, car_id, bbox, direction):
    global car_dict
    global cars_in_not_red

    if is_right_dir(direction):
        if is_in_violation_zone(bbox, video_file_name):
            if is_red(frame, video_file_name):
                if car_id in cars_in_not_red:
                    # del cars_in_not_red[np.where(cars_in_not_red == car_id)]
                    return False
                else:
                    return True
            else:
                if not car_id in cars_in_not_red:
                    cars_in_not_red = np.append(cars_in_not_red, np.array(car_id))
                    return False
                else:
                    return False
        else:
            return False
    else:
        return False
