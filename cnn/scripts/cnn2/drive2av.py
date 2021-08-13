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
from std_msgs.msg import Bool
from cv_bridge.core import CvBridge
from datetime import datetime
from tensorflow.keras.models import load_model
import pathlib
import os
import string
from std_msgs.msg import Float32

global img_rbg
global bridge
global begin_img


def preProcess(img):
    # Define Region of intrest- Perguntar ao Daniel se corto ou não , problema do angulo da camera
    #img = img[60:135, :, :]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (320, 160))
    img = np.expand_dims(img, axis=2)
    img = img/255
    return img


def message_RGB_ReceivedCallback(message):
    global img_rbg
    global bridge
    global begin_img

    img_rbg = bridge.imgmsg_to_cv2(message, "bgr8")

    begin_img = True


def signal_Callback(message):
    global vel

    vel = message.data


def main():

    # Global variables
    global vel
    global img_rbg
    global bridge
    global begin_img
    begin_img = False

    twist = Twist()

    # Init Node
    rospy.init_node('driving', anonymous=False)

    image_raw_topic = rospy.get_param('~image_raw_topic', '/ackermann_vehicle/camera/rgb/image_raw') 
    twist_cmd_topic = rospy.get_param('~twist_cmd_topic', '/cmd_vel') 
    dir_cmd_topic = rospy.get_param('~dir_cmd_topic', '/cmd_vel') 
    vel_cmd_topic = rospy.get_param('~vel_cmd_topic', '/cmd_vel') 
    float_cmd_topic = rospy.get_param('~float_cmd_topic', '') 
    modelname = rospy.get_param('~modelname', 'model1.h5')

    s = str(pathlib.Path(__file__).parent.absolute())
    path = s + '/../../models/cnn2av_' + modelname

    rospy.loginfo('Using model: %s', path)
    model = load_model(path)

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(dir_cmd_topic)
    print(vel_cmd_topic)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    # Subscribe and publish topics
    rospy.Subscriber(image_raw_topic, Image, message_RGB_ReceivedCallback)
    
    # does we need to check float_cmd_topic ?
    if float_cmd_topic != '':
        vel = 0
        rospy.Subscriber(float_cmd_topic, Float32, signal_Callback)
    else:
        vel = 1

    if twist_cmd_topic == "":
        pub_dir = rospy.Publisher(dir_cmd_topic, Twist, queue_size=10)
        pub_vel = rospy.Publisher(vel_cmd_topic, Twist, queue_size=10)
    else:
        pub_twist = rospy.Publisher(twist_cmd_topic, Twist, queue_size=10)

    # Create an object of the CvBridge class
    bridge = CvBridge()

    #Frames per second 
    rate = rospy.Rate(30)

    while not rospy.is_shutdown():

        if begin_img == False:
            continue

        resized_ = preProcess(img_rbg)

        cv2.imshow('Robot View Processed', resized_)
        cv2.imshow('Robot View', img_rbg)
        cv2.waitKey(1)

        # Predict angle & velocity
        image = np.array([resized_])
        angle = float(model.predict(image)[0][0])
        velocity = float(model.predict(image)[0][1])

        # Send twist
        if vel > 0:
            twist.linear.x = velocity
            twist.angular.z = angle
        else:
            twist.linear.x = 0
            twist.angular.z = 0
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0

        if twist_cmd_topic == "":
            pub_dir.publish(twist)
            pub_vel.publish(twist)
        else:
            pub_twist.publish(twist)

        rate.sleep()

if __name__ == '__main__':
    main()

