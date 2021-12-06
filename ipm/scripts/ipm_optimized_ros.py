#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
import pydevd_pycharm
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge.core import CvBridge
from lib import ipm_class_ros
from math import pi

roll: float = .0
pitch: float = .6
yaw: float = .0
x: float = .0
y: float = .0
z: float = .547


def on_change_roll(value: float):
    """
    Change the yaw variable
    """
    global roll
    # yaw = value
    roll = value


def on_change_pitch(value: float):
    """
    Change the yaw variable
    """
    global pitch
    # yaw = value
    pitch = value


def on_change_yaw(value: float):
    """
    Change the yaw variable
    """
    global yaw
    # yaw = value
    yaw = value


def on_change_x(value: float):
    """
    Change the yaw variable
    """
    global x
    # yaw = value
    x = value


def on_change_y(value: float):
    """
    Change the yaw variable
    """
    global y
    # yaw = value
    y = value


def on_change_z(value: float):
    """
    Change the z variable
    """
    global z
    # yaw = value
    z = value


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
    global roll, pitch, yaw, x, y, z

    # Defining variables
    bridge = CvBridge()

    pose = dict(X=x, Y=y, Z=z, r=roll, p=pitch, y=yaw)
    seeimage = False
    seeinfo = False

    # Init Node
    rospy.init_node('ipm', anonymous=False)

    # Defining parameters
    image_info_topic = rospy.get_param('~image_info_topic', '/ackermann_vehicle/camera/rgb/camera_info')
    image_raw_topic = rospy.get_param('~image_raw_topic', '/ackermann_vehicle/camera/rgb/image_raw')
    debug_mode = rospy.get_param('~debug_mode', 'False')

    # Debug settrace
    if bool(debug_mode):
        pydevd_pycharm.settrace('localhost', port=5005,
                                stdoutToServer=True, stderrToServer=True, suspend=False)

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

        # Defining IPM
        ipm = ipm_class_ros.IPM(height=height, width=width, K=K, pose=pose)

        # Converting to grayscale
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        # Calculating output image
        output_image = ipm.calculate_output_image(gray)

        # Showing image
        cv2.imshow('initial_image', img_rgb)
        cv2.imshow(winname='final_image', mat=output_image.astype(np.uint8))
        # Sliders parametrize each pose's parameter
        # FIXME actual sliders don't move
        # TODO check for way to accept float input or else do a workaround
        cv2.createTrackbar('roll', "final_image", 0, int(2.0*pi), on_change_roll)
        cv2.createTrackbar('pitch', "final_image", 0, int(2.0*pi), on_change_pitch)
        cv2.createTrackbar('yaw', "final_image", 0, int(2.0*pi), on_change_yaw)
        cv2.createTrackbar('x', "final_image", 0, 1, on_change_x)
        cv2.createTrackbar('y', "final_image", 0, 1, on_change_y)
        cv2.createTrackbar('z', "final_image", 0, 1, on_change_z)

        cv2.waitKey(1)


if __name__ == "__main__":
    main()
