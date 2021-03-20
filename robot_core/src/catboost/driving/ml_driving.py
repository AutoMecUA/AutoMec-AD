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

global img_rbg
global bridge
global begin_img

def message_RGB_ReceivedCallback(message):
    global img_rbg
    global bridge
    global begin_img


    img_rbg = bridge.imgmsg_to_cv2(message, "bgr8")
    print(img_rbg.shape)

    begin_img = True






def main():

    # Global variables
    global img_rbg
    global bridge
    global begin_img
    begin_img = False
    scale_percent = 25 # percent of original size        
    from catboost import CatBoostRegressor
    regressor = CatBoostRegressor()  
    regressor.load_model('catboost_file_turtle_test2')
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
        

        # Gray scale the image
        img_gray = cv2.cvtColor(img_rbg,cv2.COLOR_BGR2GRAY)

        # Binarize the image
        _, img_tresh = cv2.threshold(img_gray, 127, 1, cv2.THRESH_BINARY)

        # Resizing image
        width = int(img_tresh.shape[1] * scale_percent / 100)
        height = int(img_tresh.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img_tresh, dim, interpolation = cv2.INTER_AREA)
    

        # Tranform image in a list
        initial_array = copy.deepcopy(resized)
        final_array=np.resize(initial_array,(1,initial_array.shape[0]*initial_array.shape[1]))
        final_list = final_array.tolist()[0]




        # Predict angle

        angle = regressor.predict(np.array(final_list))


        # Send twist
        twist.linear.x = 0.25
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = angle

        pub.publish(twist)

        rate.sleep()








    


    
    


if __name__ == '__main__':
    main()