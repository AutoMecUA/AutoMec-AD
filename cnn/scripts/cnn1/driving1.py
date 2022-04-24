#!/usr/bin/env python3

# Imports
import argparse
import sys
import time

import cv2
from csv import writer
import copy
import numpy as np
import rospy
import yaml
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
    urdf = rospy.get_param('~urdf','')

    # Defining path to model
    s = str(pathlib.Path(__file__).parent.absolute())
    path = s + '/../../models/cnn1_' + modelname

    if not os.path.isfile(path + '_info.yaml'):
        have_dataset_yaml = False
        # we may allow to continue processing with default data
        rospy.logerr(f'You are running a model with no yaml.\n')
        while True:
            enter_pressed = input("Continue to use this model? [y/N]: ")
            if enter_pressed.lower() == "n" or enter_pressed.lower() == "no" or enter_pressed == "":
                rospy.loginfo("Shutting down")
                sys.exit()
            elif enter_pressed.lower() == "yes" or enter_pressed.lower() == "y":
                rospy.loginfo("Continuing the script")
                break
            else:
                rospy.loginfo("Please use a valid statement (yes/no)")
    else:
        with open(path + '_info.yaml') as file:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            info_loaded = yaml.load(file, Loader=yaml.FullLoader)
            ds_environment = info_loaded['model']['environment']

            if ds_environment == 'gazebo':
                ds_urdf = info_loaded['model']['urdf']

                if urdf == "":
                    rospy.logerr(f'You are running a simulation model in real life\n')
                    while True:
                        enter_pressed = input("Continue to use this model? [y/N]: ")
                        if enter_pressed.lower() == "n" or enter_pressed.lower() == "no" or enter_pressed == "":
                            rospy.loginfo("Shutting down")
                            sys.exit()
                        elif enter_pressed.lower() == "yes" or enter_pressed.lower() == "y":
                            rospy.loginfo("Continuing the script")
                            break
                        else:
                            rospy.loginfo("Please use a valid statement (yes/no)")
                elif urdf == ds_urdf:
                    rospy.loginfo(f'You are running with {urdf}')
                else:
                    time.sleep(5)
                    rospy.logerr(f'You are running with {urdf} instead of {ds_urdf} \n')
                    while True:

                        enter_pressed = input( "Continue to use this model? [y/N]: ")

                        if enter_pressed.lower() == "n" or enter_pressed.lower() == "no" or enter_pressed == "":
                            rospy.loginfo("Shutting down")
                            sys.exit()
                        elif enter_pressed.lower() == "yes" or enter_pressed.lower() == "y":
                            rospy.loginfo("Continuing the script")
                            break
                        else:
                            rospy.loginfo("Please use a valid statement (yes/no)")
            elif ds_environment == 'physical':

                if urdf != "":
                    time.sleep(5)
                    rospy.logerr(f'You are running a physical model in gazebo\n')
                    while True:
                        enter_pressed = input("Continue to use this model? [y/N]: ")
                        if enter_pressed.lower() == "n" or enter_pressed.lower() == "no" or enter_pressed == "":
                            rospy.loginfo("Shutting down")
                            sys.exit()
                        elif enter_pressed.lower() == "yes" or enter_pressed.lower() == "y":
                            rospy.loginfo("Continuing the script")
                            break
                        else:
                            rospy.loginfo("Please use a valid statement (yes/no)")
            else:
                rospy.logerr("No valid environment, please verify your YAML file")
                sys.exit()




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
