import numpy as np
import configparser
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import applications.roi.direction_detection.draw_custom_roi_wd as wd

arrow_head_xy_first = 0
arrow_tail_xy_first = 0
arrow_head_xy_second = 0
arrow_tail_xy_second = 0

polygon_first = None
polygon_second = None
total_roads = 1
DIVIDING_FACTOR = 4

roi_points = []
roi_arrow_points = []
read_config_file_points = False
output_directory_created = False

real_time_tracked_points = np.zeros((1, 3))
car_ids_and_violations = np.zeros((1, 2))


# Function Name: init_set_roi
# Description: Calls define_roi method from draw_custom_roi_wd module
# Input: (video_path: Path to input video)
# Output: Saves config.ini in path: realtime_car_analytics/data/<video name>/config/
def init_set_roi(video_path):
    """ Read one frame and call define_roi method """
    wd.define_roi(video_path)


# Function Name: get_roi_points
# Description: Get roi points from config.ini file and return all the points in a list
# Input: (select_polygon: -1 for bottom polygon | 0 for roi | 1 for top polygon)
# Output: Returns list of polygon points
def get_roi_points(config, roads):
    """ Get roi points from config.ini file and return all the points in a list """
    boundary_points = []

    if roads > 1:
        lower_left = eval(config['lower_left_second'])
        upper_left = eval(config['upper_left_second'])
        upper_right = eval(config['upper_right_second'])
        lower_right = eval(config['lower_right_second'])
    else:
        lower_left = eval(config['lower_left_first'])
        upper_left = eval(config['upper_left_first'])
        upper_right = eval(config['upper_right_first'])
        lower_right = eval(config['lower_right_first'])

    boundary_points.append(lower_left)
    boundary_points.append(upper_left)
    boundary_points.append(upper_right)
    boundary_points.append(lower_right)
    return boundary_points


# Function Name: read_config_file
# Description: Read config.ini and store all the parameters in global variables
# Input: (input_video_file: Input video name)
# Output: None. Reads config.ini and store all the parameters in global variables
def read_config_file(input_video_file):
    global arrow_head_xy_first, arrow_tail_xy_first, polygon_first
    global arrow_head_xy_second, arrow_tail_xy_second, polygon_second
    global roi_points, roi_arrow_points, read_config_file_points, total_roads

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read("./data/" + input_video_file + "/config/" + "/config.ini")
    coordinates = config['DD']

    total_roads = int(coordinates['road'])
    arrow_head_xy_first = eval(coordinates['arrow_head_xy_first'])
    arrow_tail_xy_first = eval(coordinates['arrow_tail_xy_first'])
    boundary_points_first = get_roi_points(coordinates, 1)
    polygon_first = Polygon(boundary_points_first)
    roi_points = boundary_points_first

    roi_arrow_points.append((wd.set_boundary_points_inner
                            (arrow_tail_xy_first, arrow_head_xy_first)))
    roi_arrow_points.append((wd.set_boundary_points_inner
                            (arrow_head_xy_first, arrow_tail_xy_first)))

    if total_roads > 1:
        arrow_head_xy_second = eval(coordinates['arrow_head_xy_second'])
        arrow_tail_xy_second = eval(coordinates['arrow_tail_xy_second'])
        boundary_points_second = get_roi_points(coordinates, 2)
        polygon_second = Polygon(boundary_points_second)
        roi_points = roi_points + boundary_points_second

        roi_arrow_points.append(wd.set_boundary_points_inner
                                (arrow_tail_xy_second, arrow_head_xy_second))
        roi_arrow_points.append((wd.set_boundary_points_inner
                                 (arrow_head_xy_second, arrow_tail_xy_second)))
    read_config_file_points = True


# Function Name: get_centroid
# Description: Calculate centroid of a bounding box which consists of (x1,y1,x2,y2)
# Input: (bounding_box: numpy list of x1, y1, x2, y2)
# Output: Returns (x, y) coordinate of centroid
def get_centroid(bounding_box):
    """ Calculate centroid of a bounding box which consists of (x,y,w,h) """
    # find centroid
    reshape_bbox = np.reshape(bounding_box, (2, 2))
    reshape_bbox[1] = (reshape_bbox[1] - reshape_bbox[0]) / 2
    centroid = np.sum(reshape_bbox, axis=0)
    return centroid


# Function name : get_right_wrong
# Description : Compares the direction of arrow head(P) to that of the car(Q) to find the direction of car
# Input Parameters : (P: arrow direction, Q: car direction)
# Output Parameters : Returns right direction(0) or wrong direction(1)
def get_right_wrong(P, Q):
    if (P > 0 and Q > 0) or (P < 0 and Q < 0):
        # print 'right'
        return 0
    elif (P < 0 < Q) or (P > 0 > Q):
        # print 'wrong'
        return 1


# Function Name: draw_direction_roi
# Description: Draw roi on a given frame
# Input: (frame: Image on which roi should be drawn)
# Output: None. Draws roi on a given frame
def draw_direction_roi(frame):
    for position in range(0, 3):
        wd.draw_line(frame, roi_points[position], roi_points[position+1], (3, 255, 118), 2)
    wd.draw_line(frame, roi_points[3], roi_points[0], (3, 255, 118), 3)
    if total_roads > 1:
        for position in range(4, 7):
            wd.draw_line(frame, roi_points[position], roi_points[position + 1], (3, 255, 118), 2)
        wd.draw_line(frame, roi_points[7], roi_points[4], (3, 255, 118), 3)


# Function Name: draw_arrow
# Description: Draw arrow on a given frame
# Input: (frame: image on which roi should be drawn, roads: total roads considered for direction detection (1 or 2))
# Output: None. Draws arrow on a given frame
def draw_arrow_point_in_roi(frame, roads):
    wd.draw_arrow_point(frame, roi_arrow_points[0], roi_arrow_points[1], (0, 255, 255), 2)
    if roads > 1:
        wd.draw_arrow_point(frame, roi_arrow_points[2], roi_arrow_points[3], (0, 255, 255), 2)


# Function Name: __find_direction
# Description: private, internal function to aggregate car_id points to determine direction
# Input: (frame_id, id_num: car id, cxy: centroid of bbox, arrow_direction_point: direction of arrow head)
# Output: direction of the car
#   return 0 : no direction violation
#   return 1 : direction violation
#   return -1: direction unknown
def __find_direction(frame_id, id_num, cxy, arrow_direction_point):

    global real_time_tracked_points, car_ids_and_violations
    global output_directory_created

    object_id_indices = np.where(real_time_tracked_points[:, 1] == id_num)[0]

    if object_id_indices.size == 0:
        if len(real_time_tracked_points) == 1:
            real_time_tracked_points[0] = frame_id, id_num, cxy[1]
            real_time_tracked_points = np.vstack([real_time_tracked_points, np.zeros((1, 3))])
            return -1

        real_time_tracked_points = np.vstack([real_time_tracked_points, (frame_id, id_num, cxy[1])])
        real_time_tracked_points = np.vstack([real_time_tracked_points, np.zeros((1, 3))])
        return -1

    if object_id_indices.size == 1:

        real_time_tracked_points[object_id_indices[0] + 1] = frame_id, id_num, cxy[1]
        car_id_indices = np.where(car_ids_and_violations[:, 0] == id_num)[0]

        if car_id_indices.size > 0 and (car_ids_and_violations[car_id_indices[0], 1] >= 1):
            return 1

        elif car_id_indices.size > 0 and (car_ids_and_violations[car_id_indices[0], 1] == 0):
            return 0

        return -1

    if object_id_indices.size == 2:
        real_time_tracked_points[object_id_indices[1]] = frame_id, id_num, cxy[1]
        row_subtract = real_time_tracked_points[object_id_indices[1]] - \
                       real_time_tracked_points[object_id_indices[0]]

        direction = get_right_wrong(arrow_direction_point, row_subtract[2])

        real_time_tracked_points[object_id_indices[0]] = real_time_tracked_points[object_id_indices[1]]
        real_time_tracked_points[object_id_indices[1]] = 0

        if direction == 1:

            if not np.any(car_ids_and_violations[:, 0]):
                car_ids_and_violations[0] = id_num, 1
            else:
                car_id_indices = np.where(car_ids_and_violations[:, 0] == id_num)[0]
                if car_id_indices.size == 0:
                    car_ids_and_violations = np.vstack([car_ids_and_violations, (id_num, 1)])
                else:
                    number_of_violations_of_id = car_ids_and_violations[car_id_indices[0], 1] + 1
                    car_ids_and_violations[car_id_indices[0], 1] = number_of_violations_of_id

        elif direction == 0:
            if not np.any(car_ids_and_violations[:, 0]):
                car_ids_and_violations[0] = id_num, 0
            else:
                car_id_indices = np.where(car_ids_and_violations[:, 0] == id_num)[0]
                if car_id_indices.size == 0:
                    car_ids_and_violations = np.vstack([car_ids_and_violations, (id_num, 0)])

        return direction


# Function Name: get_direction
# Description: Detect direction based on our direction detection algorithm
# Input: (video_file_name: input video name, frame, frame_id: frame number, id_num: car id,
#        bbox: predicted bounding box points)
# Output: direction of the car
#   return 0 : no direction violation
#   return 1 : direction violation
#   return -1: direction unknown
def get_direction(video_file_name, frame, frame_id, id_num, bbox):
    # print '\n data receiving in dd func', frame_id, id_num, bbox
    global car_ids_and_violations, arrow_head_xy_first, arrow_tail_xy_first
    global arrow_head_xy_second, arrow_tail_xy_second, real_time_tracked_points

    if not read_config_file_points:
        read_config_file(video_file_name)

    draw_direction_roi(frame)
    draw_arrow_point_in_roi(frame, total_roads)
    cxy = get_centroid(bbox)

    if polygon_first.contains(Point(cxy)):
        direction_of_id = __find_direction(frame_id, id_num, cxy,
                         arrow_head_xy_first[1] - arrow_tail_xy_first[1])
        return direction_of_id

    elif total_roads > 1 and polygon_second.contains(Point(cxy)):
        direction_of_id = __find_direction(frame_id, id_num, cxy,
                         arrow_head_xy_second[1] - arrow_tail_xy_second[1])
        return direction_of_id

    else:
        # print 'car_ids_and_violations', car_ids_and_violations
        if np.any(car_ids_and_violations[:, 0]):

            car_id_indices = np.where(car_ids_and_violations[:, 0] == id_num)[0]
            if car_id_indices.size > 0 and (car_ids_and_violations[car_id_indices[0], 1] >= 1):
                # print 'id {} has crossed roi'.format(id_num)
                object_id_indices = np.where(real_time_tracked_points[:, 1] == id_num)[0]
                if object_id_indices.size > 0:
                    real_time_tracked_points = np.delete(real_time_tracked_points, [object_id_indices[0], object_id_indices[0]+1], axis=0)
                return 1

            elif car_id_indices.size > 0 and (car_ids_and_violations[car_id_indices[0], 1] == 0):
                # print 'id {} has crossed roi'.format(id_num)
                object_id_indices = np.where(real_time_tracked_points[:, 1] == id_num)[0]
                if object_id_indices.size > 0:
                    real_time_tracked_points = np.delete(real_time_tracked_points,
                                                         [object_id_indices[0], object_id_indices[0] + 1], axis=0)
                return 0

        return -1


