#!/usr/bin/env python3

# Imports
import copy
from functools import partial

import cv2
import numpy as np
import rospy
from std_msgs.msg import Bool
from sensor_msgs.msg._Image import Image
from cv_bridge.core import CvBridge
import pathlib
from datetime import datetime
import pandas as pd
import signal
import sys
import os
import json

global img_rbg
global bridge
global begin_img


def largestArea(mask_original):
    """
    Create a mask with the largest blob of mask_original and return its centroid coordinates
    :param mask_original: Cv2 image - Uint8
    :return mask: Cv2 image - Bool
    :return centroid: List of 2 values
    """

    mask_original = mask_original.astype(np.uint8) * 255

    # Defining maximum area and mask label
    maxArea = 0
    maxLabel = 0

    # You need to choose 4 or 8 for connectivity type
    connectivity = 4

    # Perform the operation
    output = cv2.connectedComponentsWithStats(mask_original, connectivity, cv2.CV_32S)

    # Get the results
    # The first cell is the number of labels
    num_labels = output[0]

    # The second cell is the label matrix
    labels = output[1]

    # The third cell is the stat matrix
    stats = output[2]

    # The fourth cell is the centroid matrix
    centroids = output[3]

    # For each blob, find their area and compare it to the largest one
    for label in range(1, num_labels):
        # Find area
        area = stats[label, cv2.CC_STAT_AREA]

        # If the area is larger then the max area to date, replace it
        if area > maxArea:
            maxArea = area
            maxLabel = label

    # If there are blobs, the label is different than zero
    if maxLabel != 0:
        # Create a new mask and find its centroid
        mask = labels == maxLabel
        centroid = centroids[maxLabel]
    else:
        # If there are no blobs, the mask stays the same, and there are no centroids
        mask = mask_original
        centroid = None

    return mask



def onTrackBars(_, window_name):
    """
    Function that is called continuously to get the position of the 6 trackbars created for binarizing an image.
    The function returns these positions in a dictionary and in Numpy Arrays.
    :param _: Obligatory variable from OpenCV trackbars but assigned as a silent variable because will not be used.
    :param window_name: The name of the OpenCV window from where we need to get the values of the trackbars.
    Datatype: OpenCV object
    :return: The dictionary with the limits assigned in the trackbars. Convert the dictionary to numpy arrays because
    of OpenCV and return also.
    'limits' Datatype: Dict
    'mins' Datatype: Numpy Array object
    'maxs' Datatype: Numpy Array object
    """
    # Get ranges for each channel from trackbar and assign to a dictionary
    min_b = cv2.getTrackbarPos('min B', window_name)
    max_b = cv2.getTrackbarPos('max B', window_name)
    min_g = cv2.getTrackbarPos('min G', window_name)
    max_g = cv2.getTrackbarPos('max G', window_name)
    min_r = cv2.getTrackbarPos('min R', window_name)
    max_r = cv2.getTrackbarPos('max R', window_name)

    limits = {'B': {'min': min_b, 'max': max_b},
              'G': {'min': min_g, 'max': max_g},
              'R': {'min': min_r, 'max': max_r}}

    # Convert the dict structure created before to numpy arrays, because is the structure that opencv uses it.
    mins = np.array([limits['B']['min'], limits['G']['min'], limits['R']['min']])
    maxs = np.array([limits['B']['max'], limits['G']['max'], limits['R']['max']])

    return limits, mins, maxs


def createMask(ranges_red, ranges_green, image):
    """
    Using a dictionary wth ranges, create a mask of an image respecting those ranges
    :param ranges_red: Dictionary generated in color_segmenter.py
    :param ranges_red: Dictionary generated in color_segmenter.py
    :param image: Cv2 image - UInt8
    :return mask: Cv2 image - UInt8
    """

    # Create an array for minimum and maximum values
    mins_red = np.array([ranges_red['B']['min'], ranges_red['G']['min'], ranges_red['R']['min']])
    maxs_red = np.array([ranges_red['B']['max'], ranges_red['G']['max'], ranges_red['R']['max']])

    # Create a mask using the previously created array
    mask_red = cv2.inRange(image, mins_red, maxs_red)

    # Create an array for minimum and maximum values
    mins_green = np.array([ranges_green['B']['min'], ranges_green['G']['min'], ranges_green['R']['min']])
    maxs_green = np.array([ranges_green['B']['max'], ranges_green['G']['max'], ranges_green['R']['max']])

    # Create a mask using the previously created array
    mask_green = cv2.inRange(image, mins_green, maxs_green)

    # Unite the mask
    mask = np.zeros((image.shape[0], image.shape[1]))
    mask[mask_green.astype(np.bool)] = 1
    mask[mask_red.astype(np.bool)] = 1

    return mask.astype(np.bool)


def message_RGB_ReceivedCallback(message):
    global img_rbg
    global bridge
    global begin_img

    img_rbg = bridge.imgmsg_to_cv2(message, "bgr8")

    begin_img = True


def signal_handler(sig, frame):
    global signal_log
    global log_path

    rospy.loginfo('You pressed Ctrl+C!')
    curr_time = datetime.now()
    time_str = str(curr_time.year) + '_' + str(curr_time.month) + '_' + str(curr_time.day) + '__' + str(
        curr_time.hour) + '_' + str(curr_time.minute)
    signal_log.to_csv(log_path + '/signal_log_' + time_str + '.csv', mode='a', index=False, header=False)
    sys.exit(0)


def create_image_dict(dict_images, scale_import, N_red, path):
    # Images Importation and Resizing
    Counter_Nr_Images = 0
    for name in dict_images.keys():

        # Key and Value for the Zero and Tilt Images
        images_key = '0'
        tilt1_key = 'd'
        tilt2_key = 'l'
        tilt3_key = 'r'
        images_value = cv2.imread(path + '/' + name + '.png', cv2.IMREAD_GRAYSCALE)

        # Determination of required dimensions for the Zero Image
        width = int(images_value.shape[1] * scale_import)
        height = int(images_value.shape[0] * scale_import)
        dim = (width, height)

        # Resizing the Zero Image
        images_value = cv2.resize(images_value, dim)

        # Updating the dictionary with the Key and Value of the Zero Image
        dict_images[name]['images'][images_key] = images_value

        # Define the fraction to transform
        frac = 1 / 8

        # Locate points of the signal which you want to transform
        pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        pts2 = np.float32([[width * frac, 0], [width * (1 - frac), 0], [0, height], [width, height]])
        pts3 = np.float32(
            [[width * frac, 0], [width * (1 - frac), height * frac], [0, height], [width, height * (1 - frac)]])
        pts4 = np.float32(
            [[width * frac, height * frac], [width * (1 - frac), 0], [0, height * (1 - frac)], [width, height]])

        # Transform the original images into a tilted one
        matrix1 = cv2.getPerspectiveTransform(pts1, pts2)
        matrix2 = cv2.getPerspectiveTransform(pts1, pts3)
        matrix3 = cv2.getPerspectiveTransform(pts1, pts4)
        tilt1 = cv2.warpPerspective(images_value, matrix1, dim)
        tilt2 = cv2.warpPerspective(images_value, matrix2, dim)
        tilt3 = cv2.warpPerspective(images_value, matrix3, dim)

        # Updating the dictionary with the Key and Value of the tilted Image
        dict_images[name]['images'][tilt1_key] = tilt1
        dict_images[name]['images'][tilt2_key] = tilt2
        dict_images[name]['images'][tilt3_key] = tilt3

        Counter_Nr_Images += 4

        # Piramidization of the Zero and Tilt Image, creating smaller versions of it
        for n in range(N_red):
            # Defining the keys
            images_key = str(2 * n - 1)
            tilt1_keypyr = tilt1_key + "." + str(2 * n + 1)
            tilt2_keypyr = tilt2_key + "." + str(2 * n + 1)
            tilt3_keypyr = tilt3_key + "." + str(2 * n + 1)
            images_keyh = str(2 * n)
            tilt1_keypyrh = tilt1_key + "." + str(2 * n + 2)
            tilt2_keypyrh = tilt2_key + "." + str(2 * n + 2)
            tilt3_keypyrh = tilt3_key + "." + str(2 * n + 2)

            # Creating another lair of piramidization, assuming dimensions stay the same between signals
            width = int(images_value.shape[1] * 3 / 4)
            height = int(images_value.shape[0] * 3 / 4)
            dim = (width, height)
            images_valueh = cv2.resize(images_value, dim)
            tilt1h = cv2.resize(tilt1, dim)
            tilt2h = cv2.resize(tilt2, dim)
            tilt3h = cv2.resize(tilt3, dim)

            # Pyramidization
            images_value = cv2.pyrDown(images_value)
            tilt1 = cv2.pyrDown(tilt1)
            tilt2 = cv2.pyrDown(tilt2)
            tilt3 = cv2.pyrDown(tilt3)

            # Updating the dictionary with the Key and Value
            dict_images[name]['images'][images_key] = images_value
            dict_images[name]['images'][tilt1_keypyr] = tilt1
            dict_images[name]['images'][tilt2_keypyr] = tilt2
            dict_images[name]['images'][tilt3_keypyr] = tilt3
            dict_images[name]['images'][images_keyh] = images_valueh
            dict_images[name]['images'][tilt1_keypyrh] = tilt1h
            dict_images[name]['images'][tilt2_keypyrh] = tilt2h
            dict_images[name]['images'][tilt3_keypyrh] = tilt3h
            Counter_Nr_Images += 8

    # Number of Images Created
    rospy.loginfo("Number of images: " + str(Counter_Nr_Images))

    for name in dict_images.keys():
        for key in dict_images[name]['images']:
            dict_images[name]['images'][key] = cv2.GaussianBlur(dict_images[name]['images'][key], (3, 3), 0)
    return dict_images

def main():
    global signal_log
    global log_path
    # PARAMETERS__________________________________________________________________

    # Import Parameters
    scale_import = 0.1  # The scale of the first image, related to the imported one.
    N_red = 2  # Number of piramidizations to apply to each image.

    # Font Parameters
    subtitle_offset = -10
    subtitle_2_offset = -10
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 0, 255)
    font_thickness = 2

    # Line Parameters
    line_thickness = 3

    # Detection Parameters
    scale_cap = 0.4
    detection_threshold = 0.70

    # ______________________________________________________________________________

    # Images to import and Images Info
    dict_images = {
        'pForward': {'title': 'Follow Straight Ahead', 'type': 'Panel', 'color': 'green', 'images': {}},
        'pStop': {'title': 'Stop', 'type': 'Panel', 'color': 'red', 'images': {}},
        # 'pLeft': {'title': 'Left', 'type': 'Panel', 'color': 'green', 'images': {}},
        # 'pRight': {'title': 'Right', 'type': 'Panel', 'color': 'green', 'images': {}},
        # 'pParking': {'title': 'Parking', 'type': 'Panel', 'color': 'yellow', 'images': {}},
        'pChess': {'title': 'Chess', 'type': 'Panel', 'color': 'red', 'images': {}}
    }

    # Colors dictionary
    dict_colors = {'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0), 'yellow': (0, 255, 255)}

    # Defining variables
    global img_rbg
    global bridge
    global begin_img
    begin_img = False
    velbool = False
    segment = True
    count_stop = 0
    count_start = 0
    count_max = 2

    # Init Node
    rospy.init_node('ml_driving', anonymous=False)

    # Get parameters
    image_raw_topic = rospy.get_param('~image_raw_topic', '/ackermann_vehicle/camera2/rgb/image_raw')
    signal_cmd_topic = rospy.get_param('~signal_cmd_topic', '/signal_vel')
    mask_mode = rospy.get_param('~mask_mode', 'False')

    # Create publishers
    pubbool = rospy.Publisher(signal_cmd_topic, Bool, queue_size=10)

    # Define path for .csv
    s = str(pathlib.Path(__file__).parent.absolute())
    log_path = s + '/log/'
    rospy.loginfo(log_path)

    # If the path does not exist, create it
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # Defining variables for mask mode
    if mask_mode:
        # Create windows
        window_name_1 = 'Webcam'
        cv2.namedWindow(window_name_1, cv2.WINDOW_NORMAL)
        window_name_2 = 'Segmented image'
        cv2.namedWindow(window_name_2, cv2.WINDOW_NORMAL)

        # Use partial function for the trackbars
        onTrackBars_partial = partial(onTrackBars, window_name=window_name_2)

        # Create trackbars to control the threshold of the binarization
        cv2.createTrackbar('min B', window_name_2, 0, 255, onTrackBars_partial)
        cv2.createTrackbar('max B', window_name_2, 0, 255, onTrackBars_partial)
        cv2.createTrackbar('min G', window_name_2, 0, 255, onTrackBars_partial)
        cv2.createTrackbar('max G', window_name_2, 0, 255, onTrackBars_partial)
        cv2.createTrackbar('min R', window_name_2, 0, 255, onTrackBars_partial)
        cv2.createTrackbar('max R', window_name_2, 0, 255, onTrackBars_partial)

        # Set the trackbar position to 255 for maximum trackbars
        cv2.setTrackbarPos('max B', window_name_2, 255)
        cv2.setTrackbarPos('max G', window_name_2, 255)
        cv2.setTrackbarPos('max R', window_name_2, 255)

        # Prints to make the program user friendly. Present to the user the hotkeys
        rospy.loginfo('Use the trackbars to define the threshold limits as you wish.')
        rospy.loginfo('Start capturing the webcam video.')
        rospy.loginfo('Press "g" to save the threshold limits to green')
        rospy.loginfo('Press "r" to save the threshold limits to red')
        rospy.loginfo('Press "q" to exit without saving the threshold limits')

    # Create pandas dataframe
    signal_log = pd.DataFrame(columns=['Time', 'Signal', 'Resolution'])

    # set handler on termination
    signal.signal(signal.SIGINT, signal_handler)

    # ______________________________________________________________________________

    path = str(pathlib.Path(__file__).parent.absolute())
    dict_images = create_image_dict(dict_images, scale_import, N_red, path)

    # ______________________________________________________________________________

    # Create an object of the CvBridge class
    bridge = CvBridge()

    # Subscribe and publish topics (only after CvBridge)
    rospy.Subscriber(image_raw_topic,
                     Image, message_RGB_ReceivedCallback)

    rate = rospy.Rate(30)

    while not rospy.is_shutdown():

        if begin_img == False:
            continue

        # Defining image shape
        width_frame = img_rbg.shape[1]
        height_frame = img_rbg.shape[0]
        reduced_dim = (int(width_frame * scale_cap), int(height_frame * scale_cap))

        if mask_mode:
            if segment:
                # Get an image from the camera (a frame) and show
                frame = img_rbg
                cv2.imshow(window_name_1, frame)

                # Get ranges from trackbars in dict and numpy data structures
                limits, mins, maxs = onTrackBars_partial(0)

                # Create mask using cv2.inRange. The output is still in uint8
                segmented_frame = cv2.inRange(frame, mins, maxs)

                # Show segmented image
                cv2.imshow(window_name_2, segmented_frame)  # Display the image

                key = cv2.waitKey(1)  # Wait a key to stop the program

                # Keyboard inputs to finish the cycle
                if key == ord('q'):
                    rospy.loginfo('Letter "q" pressed, exiting the program without saving limits')
                    segment = False
                    cv2.destroyAllWindows()
                elif key == ord('g'):
                    rospy.loginfo('Letter "g" pressed, saving green limits')
                    file_name = log_path + 'limits_green.json'
                    with open(file_name, 'w') as file_handle:
                        rospy.loginfo("writing dictionary with threshold limits to file " + file_name)
                        json.dump(limits, file_handle)  # 'limits' is the dictionary
                elif key == ord('r'):
                    rospy.loginfo('Letter "r" pressed, saving red limits')
                    file_name = log_path + 'limits_red.json'
                    with open(file_name, 'w') as file_handle:
                        rospy.loginfo("writing dictionary with threshold limits to file " + file_name)
                        json.dump(limits, file_handle)  # 'limits' is the dictionary
                continue

            # Defining limits
            with open(log_path + 'limits_green.json') as file_handle:
                # returns JSON object as a dictionary
                limits_green = json.load(file_handle)
            with open(log_path + 'limits_red.json') as file_handle:
                # returns JSON object as a dictionary
                limits_red = json.load(file_handle)

            # Creating mask
            mask_frame = createMask(limits_red, limits_green, img_rbg)
            mask_frame = largestArea(mask_frame)


            # Creating masked image
            img_rbg_masked = copy.deepcopy(img_rbg)
            img_rbg_masked[~mask_frame] = 0
            img = copy.deepcopy(img_rbg_masked)
        else:
            img = copy.deepcopy(img_rbg)

        # Resizing the image
        frame = cv2.resize(img, reduced_dim)

        # Converting to a grayscale frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        max_res = 0
        max_loc = 0
        max_name = ''
        max_key = ''

        # For each image:
        for name in dict_images.keys():
            for key in dict_images[name]['images']:

                matrix_res = cv2.matchTemplate(gray_frame, dict_images[name]['images'][key], cv2.TM_CCOEFF_NORMED)
                res = np.max(matrix_res)
                loc = np.where(matrix_res == res)

                if res > max_res:
                    max_res = res
                    max_loc = loc

                    max_name = name
                    max_key = key

        # Write log files
        curr_time = datetime.now()
        time_str = str(curr_time.year) + '_' + str(curr_time.month) + '_' + str(curr_time.day) + '__' + str(
            curr_time.hour) + '_' + str(curr_time.minute) + '_' + str(curr_time.second) + '__' + str(
            curr_time.microsecond)
        # add image, angle and velocity to the signal_log pandas
        max_res_round = round(max_res, 3)
        # rospy.loginfo(max_res_round)
        row = pd.DataFrame([[time_str, max_name, max_res_round]], columns=['Time', 'Signal', 'Resolution'])
        signal_log = signal_log.append(row, ignore_index=True)

        if max_res > detection_threshold:

            max_width = int(dict_images[max_name]['images'][max_key].shape[1] / scale_cap)
            max_height = int(dict_images[max_name]['images'][max_key].shape[0] / scale_cap)

            for pt in zip(*max_loc[::-1]):
                pt = tuple(int(pti / scale_cap) for pti in pt)
                cv2.rectangle(img, pt, (pt[0] + max_width, pt[1] + max_height),
                              dict_colors.get(dict_images[max_name]['color']), line_thickness)
                text = 'Detected: ' + max_name + ' ' + max_key + ' > ' + dict_images[max_name]['type'] + ': ' + \
                       dict_images[max_name]['title']

                origin = (pt[0], pt[1] + subtitle_offset)
                origin_2 = (0, height_frame + subtitle_2_offset)
                # Using cv2.putText() method
                subtitle = cv2.putText(img, str(max_name) + '_' + str(max_key) + ' ' + str(round(max_res, 2)), origin,
                                       font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                subtitle_2 = cv2.putText(img, text, origin_2, font, font_scale, font_color, font_thickness,
                                         cv2.LINE_AA)

            # Defining and publishing the velocity of the car in regards to the signal seen
            if max_name == "pForward":
                velbool = True
                count_start = count_start + 1
                count_stop = 0
            elif max_name == "pStop":
                velbool = False
                count_stop = count_stop + 1
                count_start = 0
            elif max_name == "pChess":
                velbool = False
                count_stop = count_stop + 1
                count_start = 0
                rospy.loginfo('You have reached the end')

            if count_stop >= count_max or count_start >= count_max:
                pubbool.publish(velbool)

        else:
            count_stop = 0
            count_start = 0

        # Show image
        cv2.imshow("Frame", img_rbg)
        cv2.imshow("Frame Detections", img)
        key = cv2.waitKey(1)

        rate.sleep()

if __name__ == '__main__':
    main()