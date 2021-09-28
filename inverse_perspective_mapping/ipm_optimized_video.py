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
from ipm_optimized import IPM
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

    if begin:

        img_rbg = bridge.imgmsg_to_cv2(message, "bgr8")

        print('in')
        gray = cv2.cvtColor(img_rbg, cv2.COLOR_BGR2GRAY)

        output_image = ipm.calculate_output_image(gray)

        cv2.imshow('initial_image', img_rbg)
        cv2.imshow('final_image', output_image.astype(np.uint8))
        cv2.waitKey(0)





def main():

    global ipm
    global bridge
    global begin
    begin = False

    # Init Node
    rospy.init_node('ipm', anonymous=False)

    #rospy.Subscriber('/ackermann_vehicle/camera/rgb/image_raw', Image, message_RGB_ReceivedCallback)

    dim = (480,680)

    config_intrinsic = {'fov_x' : 1.09,
                        'fov_y' : 1.09,
                        'img_dim' : dim}

    bridge = CvBridge()

    config_extrinsic = {'camera_height' : 0.547,
                        'yaw' : 0.6 }
    
    ipm = IPM(config_intrinsic,config_extrinsic)

    cap = cv2.VideoCapture(0)

    while True:
          
        ret, frame = cap.read()
        gray = cv2.resize(frame, (dim[1],dim[0]))
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        
        print(gray.shape)
     
        output_image = ipm.calculate_output_image(gray.astype(np.uint8))

        cv2.imshow('initial_image', frame)
        cv2.imshow('final_image', output_image.astype(np.uint8))
        cv2.waitKey(0)


  


    rospy.spin()



if __name__ == "__main__":
    main()