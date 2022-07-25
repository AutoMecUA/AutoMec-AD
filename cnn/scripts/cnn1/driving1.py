#!/usr/bin/env python3

# Imports
import argparse
import sys
import time
from functools import partial
from typing import Any

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


def preProcess(img):
    # Define Region of interest
    #img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = img[40:, :]  #cut the 40 first lines
    img = cv2.resize(img, (320, 160))
    img = img/255
    return img


def message_RGB_ReceivedCallback(message, **kwargs):

    kwargs["img_rbg"] = kwargs["bridge"].imgmsg_to_cv2(message, "bgr8")

    kwargs["begin_img"] = True


def signalCallback(message, **kwargs):

    # If we receive a positive message from the signal topic, we should go. If not, we should stop.
    if message.data:
        kwargs["vel"] = kwargs["twist_linear_x"]
        kwargs["velbool"] = True
    else:
        kwargs["vel"] = 0
        kwargs["velbool"] = False


def main():

    # Global variables
    kwargs: dict[str, Any] = dict(vel=None, velbool=None, img_rbg=None,
                                  bridge=None, begin_img=None, twist_linear_x=None)

    # Defining starting values
    kwargs["begin_img"] = False
    kwargs["vel"] = 0
    kwargs["velbool"] = False
    twist = Twist()

    # Init Node
    rospy.init_node('driving', anonymous=False)

    # Getting parameters
    image_raw_topic = rospy.get_param('~image_raw_topic', '/ackermann_vehicle/camera/rgb/image_raw')
    twist_cmd_topic = rospy.get_param('~twist_cmd_topic', '')
    vel_cmd_topic = rospy.get_param('~vel_cmd_topic', '')
    kwargs["twist_linear_x"] = rospy.get_param('~twist_linear_x', 1)
    signal_cmd_topic = rospy.get_param('~signal_cmd_topic', '')
    modelname = rospy.get_param('~modelname', 'model1.h5')
    urdf = rospy.get_param('~urdf','')

    # Defining path to model
    s = str(pathlib.Path(__file__).parent.absolute())
    path = f'{s}/../../models/cnn1_{modelname}'

    if not os.path.isfile(f'{path}_info.yaml'):
        have_dataset_yaml = False
        _extracted_from_main_32('You are running a model with no yaml.\n')
    else:
        with open(f'{path}_info.yaml') as file:
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
                        elif enter_pressed.lower() in ["yes", "y"]:
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
                        elif enter_pressed.lower() in ["yes", "y"]:
                            rospy.loginfo("Continuing the script")
                            break
                        else:
                            rospy.loginfo("Please use a valid statement (yes/no)")
            elif ds_environment == 'physical':

                if urdf != "":
                    _extracted_from_main_32('You are running a physical model in gazebo\n')
            else:
                rospy.logerr("No valid environment, please verify your YAML file")
                sys.exit()

    rospy.loginfo('Using model: %s', path)
    model = load_model(path)

    # Partials
    message_RGB_ReceivedCallback_part = partial(message_RGB_ReceivedCallback, **kwargs)
    signalCallback_part = partial(signalCallback, **kwargs)

    # Subscribe and publish topics
    rospy.Subscriber(image_raw_topic, Image, message_RGB_ReceivedCallback_part)

    # If there is the signal topic, it should subscribe to it and act accordingly.
    # If not, the velocity should be max.
    if signal_cmd_topic == '':
        kwargs["vel"] = kwargs["twist_linear_x"]
        kwargs["velbool"] = True
    else:
        rospy.Subscriber(signal_cmd_topic, Bool, signalCallback_part)

    # Differentiation between gazebo and real car
    if twist_cmd_topic != '':
        pub = rospy.Publisher(twist_cmd_topic, Twist, queue_size=10)

    if vel_cmd_topic != '':
        pub_velocity = rospy.Publisher(vel_cmd_topic, Bool, queue_size=10)

    # Create an object of the CvBridge class
    kwargs["bridge"] = CvBridge()

    # Frames per second
    rate = rospy.Rate(30)

    # Timeout
    timeout_time = 4*28
    waiting = True
    start_time = None

    while not rospy.is_shutdown():

        if kwargs["begin_img"] is False:
            continue

        if waiting and kwargs["velbool"]:
            start_time = time.time()
            waiting = False

        resized_ = preProcess(kwargs["img_rbg"])

        cv2.imshow('Robot View Processed', resized_)
        cv2.imshow('Robot View', kwargs["img_rbg"])
        cv2.waitKey(1)

        # Predict angle
        image = np.array([resized_])
        steering = float(model.predict(image))
        angle = steering

        # Send twist
        twist.linear.x = kwargs["vel"]
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = angle

        # Current time
        if not waiting:
            current_time = time.time()
            time_elapsed = current_time - start_time
            rospy.loginfo(f'elapsed: {time_elapsed}')

            if current_time - start_time > timeout_time:
                pub_velocity.publish(False) 
                sys.exit()

        # To avoid any errors
        if twist_cmd_topic != '':
            pub.publish(twist)

        if vel_cmd_topic != '':
            pub_velocity.publish(kwargs["velbool"])

        rate.sleep()


# TODO Rename this here and in `main`
def _extracted_from_main_32(arg0):
    # we may allow to continue processing with default data
    time.sleep(5)
    rospy.logerr(f'{arg0}')
    while True:
        enter_pressed = input("Continue to use this model? [y/N]: ")
        if enter_pressed.lower() == "n" or enter_pressed.lower() == "no" or enter_pressed == "":
            rospy.loginfo("Shutting down")
            sys.exit()
        elif enter_pressed.lower() in ["yes", "y"]:
            rospy.loginfo("Continuing the script")
            break
        else:
            rospy.loginfo("Please use a valid statement (yes/no)")


if __name__ == '__main__':
    main()
