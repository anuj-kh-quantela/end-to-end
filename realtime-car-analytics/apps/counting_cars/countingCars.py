import numpy as np

# defining global
firstFrame = True
car_count = 0

# Function Name: count
# Description: count number of cars which crosses
# Input: one car details with frame id, track id, bounding box
# Output: counted number
def count(frame_id, id_num, bbox, h):

    global firstFrame, data_previous_frame, car_count

    # drawing a line
    line = int(0.5 * h) # 50% of the height

    # Storing present frame details as previous frame details for the first frame
    if firstFrame:
        data = np.array([[frame_id, id_num, bbox[0], bbox[1], bbox[2], bbox[3]]])
        data_previous_frame = data
        firstFrame = False

    else:

        # filtering only previous frame details by preserving present frame details for next irritation
        data_previous_frame = data_previous_frame[data_previous_frame[:, 0] >= (frame_id - 1)]

        # checking whether car is detected in previous frame or not
        if id_num in data_previous_frame[:, 1]:

            # storing present bbox of the car
            present_box = [bbox[0], bbox[1], bbox[2], bbox[3]]

            # loading previous frame details of the same car
            previous_box_id_num = data_previous_frame[
                data_previous_frame[:, 1] == id_num
            ]

            # extracting only bbox of the car
            previous_box = previous_box_id_num[0][2:6]

            # calculating the centroid of the car in both present and previous frame
            centroid_present = (present_box[0] + ((present_box[2] - present_box[0])/ 2), present_box[1] + ((present_box[3] - present_box[1])/ 2))
            centroid_previous = (previous_box[0] + ((previous_box[2] - previous_box[0]) / 2), previous_box[1] + ((previous_box[3] - previous_box[1])/ 2))


            if centroid_previous[1] <= line <= centroid_present[1] or centroid_previous[1] >= line >= centroid_present[1]:
                car_count = car_count + 1
                # print "counts ", car_count, " ", id_num
                # print "centroid_previous ", centroid_previous
                # print "centroid_present ", centroid_present

        # appending present frame details as previous frame details for next frame
        data_previous_frame = np.append(data_previous_frame, [[frame_id, id_num, bbox[0], bbox[1], bbox[2], bbox[3]]], axis=0)

    return car_count, line