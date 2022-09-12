#!/usr/bin/env python3

# Imports
import copy
from functools import partial
from re import I
from typing import Tuple, List

import cv2
import numpy as np
import rospy
from std_msgs.msg import String
from sensor_msgs.msg._Image import Image
from cv_bridge.core import CvBridge
import pathlib
from datetime import datetime
import pandas as pd
import signal
import sys
import os
import json


def largestArea(mask_original):
    """
    Create a mask with the largest blob of mask_original and return its centroid coordinates
    :param mask_original: Cv2 image - Uint8
    :return mask: Cv2 image - Bool
    :return centroid: List of 2 values
    """

    mask_original = mask_original.astype(np.uint8) * 255


    # Defining maximum area and mask label
    max_area = 150

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

    # Create mask
    mask = labels == -1

    # For each blob, find their area and compare it to the largest one
    for label in range(1, num_labels):
        # Find area
        area = stats[label, cv2.CC_STAT_AREA]

        # If the area is larger then the max area to date, replace it
        if area > max_area:
            mask_temp = labels == label
            mask = np.bitwise_or(mask, mask_temp)

    return mask


def onTrackBars(_, window_name) -> Tuple[dict, np.ndarray, np.ndarray]:
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
    min_b, min_g, min_r, max_b, max_g, max_r = [
        cv2.getTrackbarPos(val, window_name) for val in ("min B", "min G", "min R", "max B", "max G", "max R")
    ]

    limits = dict(
        B=dict(min=min_b, max=max_b),
        G=dict(min=min_g, max=max_g),
        R=dict(min=min_r, max=max_r)
    )

    # Convert the dict structure created before to numpy arrays, because is the structure that opencv uses it.
    mins: np.ndarray = np.array([limits[channel]['min'] for channel in ("B", "G", "R")])
    maxs: np.ndarray = np.array([limits[channel]['max'] for channel in ("B", "G", "R")])

    return limits, mins, maxs


def createMask(ranges_red, ranges_green, image) -> np.ndarray:
    """
    Using a dictionary wth ranges, create a mask of an image respecting those ranges
    :param ranges_red: Dictionary generated in color_segmenter.py
    :param ranges_green: Dictionary generated in color_segmenter.py
    :param image: Cv2 image - UInt8
    :return mask: Cv2 image - UInt8
    """

    mask_red = colorMask(image, ranges_red)

    mask_green = colorMask(image, ranges_green)

    # Unite the mask
    mask = np.zeros((image.shape[0], image.shape[1]))
    mask[mask_green.astype(bool)] = 1
    mask[mask_red.astype(bool)] = 1

    return mask.astype(bool)


def colorMask(image, color_ranges):
    # Create an array for minimum and maximum values
    mins_red = np.array([color_ranges['B']['min'], color_ranges['G']['min'], color_ranges['R']['min']])
    maxs_red = np.array([color_ranges['B']['max'], color_ranges['G']['max'], color_ranges['R']['max']])
    # Create a mask using the previously created array
    mask_red = cv2.inRange(image, mins_red, maxs_red)
    return mask_red


def rgbMsgCallback(message, img_args):
    """Receive image

    Args:
        message (rgb msg): image in ros msg format
        img_args (dict): input and output arguments
    """

    img_args['img_rgb'] = img_args['bridge'].imgmsg_to_cv2(message, "bgr8")

    img_args['begin_img'] = True



def create_image_dict(dict_images, scale_import, n_red, path):
    # Images Importation and Resizing
    counter_nr_images: int = 0
    for name in dict_images.keys():

        # Key and Value for the Zero and Tilt Images
        images_key = '0'
        tilt1_key = 'd'
        tilt2_key = 'l'
        tilt3_key = 'r'
        images_value = cv2.imread(f"{path}/{name}.png", cv2.IMREAD_GRAYSCALE)

        # Determination of required dimensions for the Zero Image
        height, width = [int(images_value.shape[dim_id] * scale_import) for dim_id in (0, 1)]
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
        matrix1, matrix2, matrix3 = [cv2.getPerspectiveTransform(pts1, dst) for dst in (pts2, pts3, pts4)]
        tilt1, tilt2, tilt3 = [cv2.warpPerspective(images_value, matrix, dim) for matrix in (matrix1, matrix2, matrix3)]

        # Updating the dictionary with the Key and Value of the tilted Image
        dict_images[name]['images'][tilt1_key] = tilt1
        dict_images[name]['images'][tilt2_key] = tilt2
        dict_images[name]['images'][tilt3_key] = tilt3

        counter_nr_images += 4

        # Piramidization of the Zero and Tilt Image, creating smaller versions of it
        for n in range(n_red):
            # Defining the keys
            images_key = str(2 * n - 1)
            tilt1_keypyr, tilt2_keypyr, tilt3_keypyr = [
                f"{tilt_key}.{2 * n + 1}" for tilt_key in (tilt1_key, tilt2_key, tilt3_key)
            ]
            images_keyh = str(2 * n)
            tilt1_keypyrh, tilt2_keypyrh, tilt3_keypyrh = [
                f"{tilt_key}.{2 * n + 2}" for tilt_key in (tilt1_key, tilt2_key, tilt3_key)
            ]

            # Creating another lair of piramidization, assuming dimensions stay the same between signals
            height, width = [int(images_value.shape[dim_id] * 3 / 4) for dim_id in (0, 1)]
            dim = (width, height)
            images_valueh = cv2.resize(images_value, dim)
            tilt1h, tilt2h, tilt3h = [cv2.resize(tilt, dim) for tilt in (tilt1, tilt2, tilt3)]

            # Pyramidization
            images_value = cv2.pyrDown(images_value)
            tilt1, tilt2, tilt3 = [cv2.pyrDown(tilt) for tilt in (tilt1, tilt2, tilt3)]

            # Updating the dictionary with the Key and Value
            dict_images[name]['images'] = {
                images_key: images_value,
                tilt1_keypyr: tilt1,
                tilt2_keypyr: tilt2,
                tilt3_keypyr: tilt3,
                tilt1_keypyrh: tilt1h,
                tilt2_keypyrh: tilt2h,
                tilt3_keypyrh: tilt3h
            }
            counter_nr_images += 8

    # Number of Images Created
    rospy.loginfo(f"Number of images: {counter_nr_images}")

    for name in dict_images.keys():
        for key in dict_images[name]['images']:
            dict_images[name]['images'][key] = cv2.GaussianBlur(dict_images[name]['images'][key], (3, 3), 0)
    return dict_images


def main():
    # PARAMETERS__________________________________________________________________

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
    detection_threshold = 0.7

    # ______________________________________________________________________________

    # Images to import and Images Info
    dict_images = dict(
        pForward=dict(title='Follow Straight Ahead', type='Panel', color='green', images={}),
        pStop=dict(title='Stop', type='Panel', color='red', images={}),
        pLeft=dict(title='Left', type='Panel', color='green', images={}),
        pRight=dict(title='Right', type='Panel', color='green', images={}),
        pParking=dict(title='Parking', type='Panel', color='yellow', images={}),
        pChess=dict(title='Chess', type='Panel', color='red', images={}),
        pChessBlack=dict(title='ChessBlack', type='Panel', color='red', images={}),
        pChessBlackInv=dict(title='ChessBlackInv', type='Panel', color='red', images={})
    )

    # Colors dictionary
    dict_colors = dict(red=(0, 0, 255), green=(0, 255, 0), blue=(255, 0, 0), yellow=(0, 255, 255))

    # Defining variables
    img_args = dict(img_rgb = None, begin_img = False, bridge = None)
    segment = True
    detection_num_in = 4
    detection_num_out = 5
    max_name_list = [None] * detection_num_out
    N_red = 2
    scale_import = 0.2  # The scale of the first image, related to the imported one.

    # Init Node
    rospy.init_node('ml_driving', anonymous=False)

    # Get parameters
    image_raw_topic = rospy.get_param('~image_raw_topic', '/top_left_camera/image_raw')
    signal_cmd_topic = rospy.get_param('~signal_cmd_topic', '/signal_detected')
    mask_mode = rospy.get_param('~mask_mode', 'True')

    # Create publishers
    signal_pub = rospy.Publisher(signal_cmd_topic, String, queue_size=10)

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
        window_name_1 = image_raw_topic
        cv2.namedWindow(window_name_1, cv2.WINDOW_NORMAL)
        window_name_2 = 'Image to segment'
        cv2.namedWindow(window_name_2, cv2.WINDOW_NORMAL)

        # Use partial function for the trackbars
        onTrackBars_partial = partial(onTrackBars, window_name=window_name_2)

        # Create trackbars to control the threshold of the binarization
        create_track_list = ["min B", "max B", "min G", "max G", "min R", "max R"]

        for idx, param in enumerate(create_track_list):
            cv2.createTrackbar(param, window_name_2, 0, 255, onTrackBars_partial)
            
           # Set the trackbar position to 255 for maximum trackbars
            if idx % 2 == 1:
                cv2.setTrackbarPos(param, window_name_2, 255)

        # Prints to make the program user-friendly. Present to the user the hotkeys
        rospy.loginfo('Use the trackbars to define the threshold limits as you wish.')
        rospy.loginfo('Start capturing the webcam video.')
        rospy.loginfo('Press "g" to save the threshold limits to green')
        rospy.loginfo('Press "r" to save the threshold limits to red')
        rospy.loginfo('Press "q" to exit without saving the threshold limits')
    else:
        rospy.loginfo('Press "s" to exit, saving the log')
        rospy.loginfo('Press "q" to exit without saving the log')

    # Create pandas dataframe
    signal_log = pd.DataFrame(columns=['Time', 'Signal', 'Resolution'])



    path = str(pathlib.Path(__file__).parent.absolute())
    dict_images = create_image_dict(dict_images, scale_import, N_red, path)
    # ______________________________________________________________________________

    # Create an object of the CvBridge class
    img_args['bridge'] = CvBridge()

    # Subscribe and publish topics (only after CvBridge)
    rgbMsgCallback_part = partial(rgbMsgCallback, img_args=img_args)
    rospy.Subscriber(image_raw_topic,
                     Image, rgbMsgCallback_part)

    rate = rospy.Rate(30)

    while not rospy.is_shutdown():
        if img_args['begin_img'] is False:
            continue

        # Defining image shape
        img_rgb = img_args['img_rgb']
        height_frame, width_frame = [img_rgb.shape[dim_id] for dim_id in (0, 1)]
        reduced_dim = (int(width_frame * scale_cap), int(height_frame * scale_cap))

        if mask_mode:
            if segment:
                frame = img_rgb
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
                    rospy.loginfo('Letter "q" pressed, exiting the program')
                    segment = False
                    cv2.destroyAllWindows()
                    rospy.loginfo('Press "s" to exit, saving the log')
                    rospy.loginfo('Press "q" to exit without saving the log')
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
            mask_frame = createMask(limits_red, limits_green, img_rgb)
            mask_frame = largestArea(mask_frame)


            # Creating masked image
            img_rgb_masked = copy.deepcopy(img_rgb)
            img_rgb_masked[~mask_frame] = 0
            img = copy.deepcopy(img_rgb_masked)
        else:
            img = copy.deepcopy(img_rgb)

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
        time_str: str = f"{curr_time.year}_{curr_time.month}_{curr_time.day}" \
                   f"__{curr_time.hour}_{curr_time.minute}_{curr_time.second}__{curr_time.microsecond}"
        # add image, angle and velocity to the signal_log pandas
        max_res_round = round(max_res, 3)
        # rospy.loginfo(max_res_round)
        row = pd.DataFrame([[time_str, max_name, max_res_round]], columns=['Time', 'Signal', 'Resolution'])
        signal_log = pd.concat([signal_log, row])
        if max_res > detection_threshold:
            max_height, max_width = [
                int(dict_images[max_name]['images'][max_key].shape[dim_id] / scale_cap) for dim_id in (0, 1)
            ]

            for pt in zip(*max_loc[::-1]):
                pt = tuple(int(pti / scale_cap) for pti in pt)
                cv2.rectangle(img, pt, (pt[0] + max_width, pt[1] + max_height),
                              dict_colors.get(dict_images[max_name]['color']), line_thickness)
                text = f"Detected: {max_name} {max_key} > {dict_images[max_name]['type']}" \
                       f": {dict_images[max_name]['title']}"

                origin = (pt[0], pt[1] + subtitle_offset)
                origin_2 = (0, height_frame + subtitle_2_offset)
                # Using cv2.putText() method
                subtitle = cv2.putText(img, f"{max_name}_{max_key} {round(max_res, 2)}", origin,
                                       font, font_scale, font_color, font_thickness, cv2.LINE_AA)
                subtitle_2 = cv2.putText(img, text, origin_2, font, font_scale, font_color, font_thickness,
                                         cv2.LINE_AA)
            
            # Append name of detected signal to the list
            max_name_list.append(max_name)
            
        else:
            max_name_list.append(None)

        # If the length of the list is larger than the predetermined one, erase the oldest element of said list
        while len(max_name_list) > detection_num_out:
            max_name_list.pop(0)

        # If the number of detections of a certain signal is larger than predetermined, publish
        for name in dict_images.keys():
            mask_list = [int(detection == name) for detection in max_name_list]
            if sum(mask_list) >= detection_num_in:
                signal_pub.publish(name)

        # Show image
        cv2.imshow("Frame", img_rgb)
        cv2.imshow("Frame Detections", img)
        key = cv2.waitKey(1)

        if key == ord('s'):
            rospy.loginfo('Letter "s" pressed, exiting the program while saving the log')
            cv2.destroyAllWindows()
            curr_time = datetime.now()
            time_str = f"{curr_time.year}_{curr_time.month}_{curr_time.day}__{curr_time.hour}_{curr_time.minute}"
            signal_log.to_csv(f"{log_path}/signal_log_{time_str}.csv", mode='a', index=False, header=False)
            rospy.signal_shutdown('Manual shutdown')
        elif key == ord('q'):
            rospy.loginfo('Letter "q" pressed, exiting the program')
            cv2.destroyAllWindows()
            rospy.signal_shutdown('Manual shutdown')

        rate.sleep()


if __name__ == '__main__':
    main()