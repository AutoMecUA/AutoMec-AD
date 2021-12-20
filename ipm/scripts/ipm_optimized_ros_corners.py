#!/usr/bin/env python3
import copy
from functools import partial

import cv2
import numpy as np
import rospy
import pydevd_pycharm
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge.core import CvBridge
from typing import Tuple, Dict

from lib import ipm_class_ros
from math import pi

# Get debug_mode param
debug_mode = rospy.get_param('~debug_mode', 'False')
# Debug settrace
if bool(debug_mode):
    try:
        pydevd_pycharm.settrace('localhost', port=5005,
                                stdoutToServer=True, stderrToServer=True, suspend=False)
    except ConnectionRefusedError:
        rospy.logwarn("Didn't find any debug server (Errno 111: Connection refused). "
                      "Make sure you launch it before this script.")

# Pose parameters' scope constants (min,max)
RPY_RANGE: Tuple[float, float] = (.0, 2 * pi)
XY_RANGE: Tuple[float, float] = (-10., 10.)
Z_RANGE: Tuple[float, float] = (-3., 3.)
# Maps each parameter to its range
ID_RANGES: dict = dict(
    roll=RPY_RANGE, pitch=RPY_RANGE, yaw=RPY_RANGE,
    x=XY_RANGE, y=XY_RANGE, z=Z_RANGE
)


def normalize(range_max: float, range_min: float, percentage: int) -> float:
    """
    Maps a percentage value to its absolute given the minimum value (0%) and maximum (100%)
    """
    portion: float = percentage / 100.
    return range_min + (range_max - range_min) * portion


def onTrackBars(_, window_name) -> Dict[str, float]:
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

    # Pose parameters
    pose_params: tuple = ("roll", "pitch", "yaw", "x", "y", "z")

    pose: dict = {
        param: normalize(range_max=ID_RANGES[param][1],
                         range_min=ID_RANGES[param][0],
                         percentage=cv2.getTrackbarPos(param, window_name))
        for param in pose_params
    }

    return pose

def onMouse(event, x, y, flags, param):
    """
    Retrieving mouse coordinates while the left button is clicked
    Based on https://github.com/lucasrdalcol/PSR_Assignment2_P2_G8
    :param event:
    :param x:
    :param y:
    :param flags:
    :param param:
    :return:
    """
    global corners

    if event == cv2.EVENT_LBUTTONDOWN:
        corners.append((x, y))
        if len(corners) > 4:
            corners = corners[1:]
        print(corners)

    return

def define_corners(image, list):
    """
    Defining the corners of the track on an image
    """
    global corners
    corners = list
    window = 'Define the corners'
    cv2.namedWindow(window)
    cache = copy.deepcopy(image)
    while True:
        cv2.setMouseCallback(window, onMouse)
        if bool(corners):
            cache = copy.deepcopy(image)
            for corner in corners:
                cv2.circle(img=cache, center=corner, radius=0, color=(0, 255, 0), thickness=10)
        cv2.imshow(window, cache)
        key = cv2.waitKey(1)
        if (key & 0xFF) == ord("q"):
            if len(corners) < 4:
                rospy.logwarn('Please define 4 points, you only have ' + str(len(corners)) + ' defined')
            elif len(corners) == 4:
                rospy.loginfo('4 corners selected')
                cv2.destroyAllWindows()
                break
            else:
                rospy.loginfo(str(len(corners)) + ' points selected, only using the last 4')
                corners = corners[(len(corners)-4):]
                break
    return corners


# Callback function to receive image
def message_RGB_ReceivedCallback(message):
    # Defining global variables
    global bridge
    global ipm
    global img_rgb
    global seeimage

    # Transforming ROS image into CV2 image
    img_rgb = bridge.imgmsg_to_cv2(message, "bgr8")

    # Setting the variable as true
    seeimage = True


def message_Info_ReceivedCallback(message):
    # Defining global variables
    global K
    global height, width
    global seeinfo

# Retrieving intrinsic matrix and converting it
    K_tu = message.K
    K_norm = ([K_tu[0], K_tu[1], K_tu[2]], [K_tu[3], K_tu[4], K_tu[5]], [K_tu[6], K_tu[7], K_tu[8]])
    K = np.asarray(K_norm)

    # Retrieving heigth and width
    height = message.height
    width = message.width

    # Setting the variable as true
    seeinfo = True


def main():
    global bridge
    global K
    global height, width
    global img_rgb
    global seeimage, seeinfo
    # Pose parameters
    roll: float = 0.6
    pitch: float = .0
    yaw: float = .0
    x: float = .0
    y: float = .0
    z: float = .547


    # Defining variables
    bridge = CvBridge()
    corners = []
    pose = dict(X=x, Y=y, Z=z, r=roll, p=pitch, y=yaw)
    seeimage = False
    seeinfo = False
    window_name_1 = "Webcam Input"
    window_name_2 = "IPM output"

    # Init Node
    rospy.init_node('ipm', anonymous=False)

    # Defining parameters
    image_info_topic = rospy.get_param('~image_info_topic', '/ackermann_vehicle/camera/rgb/camera_info')
    image_raw_topic = rospy.get_param('~image_raw_topic', '/ackermann_vehicle/camera/rgb/image_raw')
    # Subscribing to both topics
    rospy.Subscriber(image_info_topic, CameraInfo, message_Info_ReceivedCallback)
    rospy.Subscriber(image_raw_topic, Image, message_RGB_ReceivedCallback)

    # Continuous running
    while not rospy.is_shutdown():

        # If it does not receive info, return to the beginning of the cycle
        if not seeinfo:
            continue

        rospy.loginfo('Received camera info')

        # If it does not receive image, return to the beginning of the cycle
        if not seeimage:
            continue

        rospy.loginfo('Received camera image')

        if not bool(corners):
            # Defining corners
            rospy.loginfo('Defining corners')
            corners = define_corners(img_rgb, corners)
            rospy.loginfo('Corners Defined')

            # Create windows
            cv2.namedWindow(window_name_1, cv2.WINDOW_NORMAL)
            cv2.namedWindow(window_name_2, cv2.WINDOW_NORMAL)

            # Defining IPM
            rospy.loginfo('Pose: ' + str(pose))
            ipm = ipm_class_ros.IPM(height=height, width=width, K=K, pose=pose)
            ipm.calculate_corners_coords(corners)

        # Converting to grayscale
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        # Calculating output image
        # output_image = ipm.calculate_output_image(gray)
        output_image = ipm.calculate_corners_image(img_in=gray)

        # Showing image
        cv2.imshow(winname=window_name_1, mat=img_rgb)
        cv2.imshow(winname=window_name_2, mat=output_image.astype(np.uint8))
        # Sliders parametrize pose's each parameter
        onTrackBars_partial: partial[dict] = partial(onTrackBars, window_name=window_name_2)
        # FIXME actual sliders don't move
        cv2.createTrackbar('roll', window_name_2, 0, 100, onTrackBars_partial)
        cv2.createTrackbar('pitch', window_name_2, 0, 100, onTrackBars_partial)
        cv2.createTrackbar('yaw', window_name_2, 0, 100, onTrackBars_partial)
        cv2.createTrackbar('x', window_name_2, 0, 100, onTrackBars_partial)
        cv2.createTrackbar('y', window_name_2, 0, 100, onTrackBars_partial)
        cv2.createTrackbar('z', window_name_2, 0, 100, onTrackBars_partial)

        cv2.waitKey(1)


if __name__ == "__main__":
    main()
