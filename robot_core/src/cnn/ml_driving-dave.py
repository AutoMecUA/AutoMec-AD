#!/usr/bin/env python3

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
import pathlib
import os
import string


global img_rbg
global bridge
global begin_img


def preProcess(img):
    # Define Region of intrest- Perguntar ao Daniel se corto ou não , problema do angulo da camera
    #img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img


def message_RGB_ReceivedCallback(message):
    global img_rbg
    global bridge
    global begin_img

    img_rbg = bridge.imgmsg_to_cv2(message, "bgr8")

    begin_img = True


def main():

    # Global variables
    global img_rbg
    global bridge
    global begin_img
    begin_img = False

    twist = Twist()

    # Init Node
    rospy.init_node('ml_driving', anonymous=False)

    image_raw_topic = rospy.get_param('~image_raw_topic', '/ackermann_vehicle/camera/rgb/image_raw') 
    twist_cmd_topic = rospy.get_param('~twist_cmd_topic', '/cmd_vel') 
    twist_linear_x = rospy.get_param('~twist_linear_x', 1.0)
    modelname = rospy.get_param('~modelname', 'model_sergio4teste.h5')

    s = str(pathlib.Path(__file__).parent.absolute())
    path = s + '/models_files/' + modelname
    print (path)
    model = load_model(path)

    # Subscribe and publish topics
    #rospy.Subscriber('/ackermann_vehicle/camera/rgb/image_raw',
    #                 Image, message_RGB_ReceivedCallback)
    #pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    #
    rospy.Subscriber(image_raw_topic,
                     Image, message_RGB_ReceivedCallback)
    pub = rospy.Publisher(twist_cmd_topic, Twist, queue_size=10)

    # Create an object of the CvBridge class
    bridge = CvBridge()

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():

        if begin_img == False:
            continue

        resized_ = preProcess(img_rbg)

        cv2.imshow('Robot View Processed', resized_)
        cv2.imshow('Robot View', img_rbg)
        cv2.waitKey(1)

        # Predict angle
        image = np.array([resized_])
        steering = float(model.predict(image))
        angle = steering

        # Send twist
        twist.linear.x = twist_linear_x
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = angle

        pub.publish(twist)

        rate.sleep()

if __name__ == '__main__':
    main()
