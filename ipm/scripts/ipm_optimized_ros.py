#!/usr/bin/env python3

# Imports
import cv2
import numpy as np
import copy
import numpy as np
from numpy.lib.function_base import angle
import rospy
from geometry_msgs.msg._Twist import Twist
from std_msgs.msg import Bool
from sensor_msgs.msg._Image import Image
from cv_bridge.core import CvBridge
from datetime import datetime
import pandas as pd
from ipm_optimized2 import IPM
from sklearn import preprocessing
global ipm
global bridge
global begin
import cv2

# Callback function to receive image
def message_RGB_ReceivedCallback(message):
    global img_rbg
    global bridge
    global begin_img
    global bridge
    global ipm
    global begin    



    img_rbg = bridge.imgmsg_to_cv2(message, "bgr8")

    print('in')
    gray = cv2.cvtColor(img_rbg, cv2.COLOR_BGR2GRAY)

    output_image = ipm.calculate_output_image(gray)

    cv2.imshow('initial_image', img_rbg)
    cv2.imshow('final_image', output_image.astype(np.uint8))
    cv2.waitKey(1)





def main():
    global ipm
    global bridge
    global begin
    begin = True

    # Init Node
    rospy.init_node('ipm', anonymous=False)

    image_raw_topic = rospy.get_param('~image_raw_topic', '/ackermann_vehicle/camera/rgb/image_raw')

    dim = (400,300)

    config_intrinsic = {'fov_x' : 1.09,
                        'fov_y' : 1.09,
                        'img_dim' : dim}

    bridge = CvBridge()

    config_extrinsic = {'camera_height': 0.547,
                        'yaw': 0.6}

    ipm = IPM(config_intrinsic, config_extrinsic)

    bridge = CvBridge()

    rate = rospy.Rate(1)

    while not rospy.is_shutdown():
        rospy.Subscriber(image_raw_topic, Image, message_RGB_ReceivedCallback)
        rate.sleep()
   
        


  


    



if __name__ == "__main__":
    main()