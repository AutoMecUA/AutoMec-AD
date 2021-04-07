#!/usr/bin/env python

# Imports
import argparse
import cv2
from csv import writer
import copy
import numpy as np
import rospy
from geometry_msgs.msg._Twist import Twist
from sensor_msgs.msg._Image import Image
from cv_bridge.core import CvBridge
from datetime import datetime
from tensorflow.keras.models import load_model

global img_rbg
global bridge
global begin_img

def message_RGB_ReceivedCallback(message):
    global img_rbg
    global bridge
    global begin_img


    img_rbg = bridge.imgmsg_to_cv2(message, "bgr8")
    

    begin_img = True






def main():

    model = load_model('model_daniel.h5')

    # Global variables
    global img_rbg
    global bridge
    global begin_img
    begin_img = False
     
    
    twist = Twist()


    # Init Node
    rospy.init_node('ml_driving', anonymous=False)

    # Subscribe topics
    rospy.Subscriber('/robot/camera/rgb/image_raw', Image, message_RGB_ReceivedCallback)
    pub = rospy.Publisher('robot/cmd_vel',Twist, queue_size=10)
    # Create an object of the CvBridge class
    bridge = CvBridge()

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():

        if begin_img==False:
            continue
        
        width = 320
        height = 160
        dim = (width, height)
        img_rbg = cv2.resize(img_rbg, dim, interpolation = cv2.INTER_AREA)



        # Predict angle

        image = np.array(img_rbg)
        angle = float(model.predict(image))


        # CAMERA PARAMETERS

        # <origin xyz="0.003 0.011 0.59" rpy="0 0.4 0"/> 



        # Send twist
        twist.linear.x = 0.40
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = angle

        pub.publish(twist)

        rate.sleep()








    


    
    


if __name__ == '__main__':
    main()