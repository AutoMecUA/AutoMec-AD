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
    # Define Region of interest
    #img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (320, 160))
    img = img/255
    return img

def message_RGB_ReceivedCallback(message):
    global img_rbg
    global bridge
    global begin_img

    img_rbg = bridge.imgmsg_to_cv2(message, "bgr8")

    begin_img = True


def signalCallback(message):
    global vel
    global velbool
    global twist_linear_x

    # If we receive a positive message from the signal topic, we should go. If not, we should stop.
    if message.data:
        vel = twist_linear_x
        velbool = True
    else:
        vel = 0
        velbool = False


def main():

    # Global variables
    global vel, velbool
    global img_rbg
    global bridge
    global begin_img
    global twist_linear_x

    # Defining starting values
    begin_img = False
    vel = 0
    velbool = False
    twist = Twist()

    # Init Node
    rospy.init_node('driving', anonymous=False)

    # Getting parameters
    image_raw_topic = rospy.get_param('~image_raw_topic', '/ackermann_vehicle/camera/rgb/image_raw')
    twist_cmd_topic = rospy.get_param('~twist_cmd_topic', '')
    vel_cmd_topic = rospy.get_param('~vel_cmd_topic', '')
    twist_linear_x = rospy.get_param('~twist_linear_x', 1)
    signal_cmd_topic = rospy.get_param('~signal_cmd_topic', '')
    modelname = rospy.get_param('~modelname', 'model1.h5')

    # Defining path to model
    s = str(pathlib.Path(__file__).parent.absolute())
    path = s + '/../../models/cnn1_' + modelname

    rospy.loginfo('Using model: %s', path)
    model = load_model(path)

    # Subscribe and publish topics
    rospy.Subscriber(image_raw_topic, Image, message_RGB_ReceivedCallback)

    # If there is the signal topic, it should subscribe to it and act accordingly.
    # If not, the velocity should be max.
    if signal_cmd_topic == '':
        vel = twist_linear_x
        velbool = True
    else:
        rospy.Subscriber(signal_cmd_topic, Bool, signalCallback)

    # Differentiation between gazebo and real car
    if twist_cmd_topic != '':
        pub = rospy.Publisher(twist_cmd_topic, Twist, queue_size=10)

    if vel_cmd_topic != '':
        pub_velocity = rospy.Publisher(vel_cmd_topic, Bool, queue_size=10)


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

        # Predict angle
        image = np.array([resized_])
        steering = float(model.predict(image))
        angle = steering

        # Send twist
        twist.linear.x = vel
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = angle

        # To avoid any errors
        if twist_cmd_topic != '':
            pub.publish(twist)

        if vel_cmd_topic != '':
            pub_velocity.publish(velbool)

        rate.sleep()

if __name__ == '__main__':
    main()
