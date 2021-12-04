#!/usr/bin/env python3

# Imports
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge.core import CvBridge
from lib import ipm_class_ros

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

    # Defining variables
    bridge = CvBridge()
    cam_height = 0.547
    yaw = 0.6
    seeimage = False
    seeinfo = False

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

        # Defining IPM
        ipm = ipm_class_ros.IPM(height=height, width=width, K=K, yaw=yaw, cam_height=cam_height)

        # Converting to grayscale
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

        # Calculating output image
        output_image = ipm.calculate_output_image(gray)

        # Showing image
        cv2.imshow('initial_image', img_rgb)
        cv2.imshow('final_image', output_image.astype(np.uint8))
        cv2.waitKey(1)



if __name__ == "__main__":
    main()
