#!/usr/bin/env python3

# Imports
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge.core import CvBridge
from lib import ipm_class_ros

global ipm
global bridge
global begin


# Callback function to receive image
def message_RGB_ReceivedCallback(message):
    # Defining global variables
    global bridge
    global ipm
    global img_rgb
    global output_image
    global seeimage

    # Transforming ROS image into CV2 image
    img_rgb = bridge.imgmsg_to_cv2(message, "bgr8")

    # Converting to grayscale
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    # Calculating output image
    output_image = ipm.calculate_output_image(gray)

    seeimage = True



def message_Info_ReceivedCallback(message):
    # Defining global variables
    global K
    global R
    global height, width
    global seeinfo

    K = message.K
    R = message.R
    height = message.height
    width = message.width
    seeinfo = True


def main():
    global bridge
    global K
    global R
    global height, width
    global ipm
    global seeimage, seeinfo

    # Defining variables
    bridge = CvBridge()
    cam_height = 0.547
    seeimage = False
    seeinfo = False

    # Init Node
    rospy.init_node('ipm', anonymous=False)

    # Defining parameters
    image_info_topic = rospy.get_param('~image_info_topic', '/ackermann_vehicle/camera/rgb/camera_info')
    image_raw_topic = rospy.get_param('~image_raw_topic', '/ackermann_vehicle/camera/rgb/image_raw')

    rospy.Subscriber(image_info_topic, CameraInfo, message_Info_ReceivedCallback)

    # print(seeinfo)

    if seeinfo:
        rospy.loginfo('Received camera info')
        ipm = ipm_class_ros.IPM(height=height, width=width, K=K, R=R, cam_height=cam_height)
        rospy.Subscriber(image_raw_topic, Image, message_RGB_ReceivedCallback)

        if seeimage:
            rospy.loginfo('Received camera image')
            cv2.imshow('initial_image', img_rgb)
            cv2.imshow('final_image', output_image.astype(np.uint8))
            cv2.waitKey(1)

    rospy.spin()


if __name__ == "__main__":
    main()
