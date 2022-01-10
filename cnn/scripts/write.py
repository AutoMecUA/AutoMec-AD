#!/usr/bin/env python3

# Imports
import os
import signal
import sys
import cv2
from csv import writer
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
import datetime
from PIL import Image as Image_pil

import pathlib

import yaml

# Global Variables
global angular
global linear
global bridge
global begin_cmd
global begin_img
global img_rbg
global driving_log
global data_path


# Calback Function to receive the cmd values
def messageReceivedCallback(message):
    global angular
    global linear
    global begin_cmd

    angular = float(message.angular.z)
    linear = float(message.linear.x)

    begin_cmd = True


def messageRealReceivedCallback(message):
    global angular

    angular = float(message.angular.z)



def boolReceivedCallback(message):
    global linear
    global begin_cmd

    if message.data:
        linear = 1
        begin_cmd = True
    else:
        linear = 0
        begin_cmd = False


# Callback function to receive image
def message_RGB_ReceivedCallback(message):
    global img_rbg
    global bridge
    global begin_img

    img_rbg = bridge.imgmsg_to_cv2(message, "bgr8")

    begin_img = True


def signal_handler(sig, frame):
    global driving_log
    global data_path

    rospy.loginfo('You pressed Ctrl+C!')
    driving_log.to_csv(data_path + '/driving_log.csv', mode='a', index=False, header=False)
    sys.exit(0)


def main():
    # Global variables
    global angular
    global linear
    global bridge
    global img_rbg
    global begin_cmd
    global begin_img
    global driving_log

    global data_path

    # Initial Value
    begin_cmd = False
    begin_img = False
    first_time = True

    # Init Node
    rospy.init_node('write_data', anonymous=False)

    image_raw_topic = rospy.get_param('~image_raw_topic', '/ackermann_vehicle/camera/rgb/image_raw')
    twist_cmd_topic = rospy.get_param('~twist_cmd_topic', '/cmd_vel')
    vel_cmd_topic = rospy.get_param('~vel_cmd_topic', '')
    base_folder = rospy.get_param('~folder', 'set1')
    rate_hz = rospy.get_param('~rate', 30)
    image_width = rospy.get_param('~width', 320)
    image_height = rospy.get_param('~height', 160)

    # params only used in yaml file
    cam_height = rospy.get_param('~cam_height', '')
    cam_angle = rospy.get_param('~cam_angle', '')
    env = rospy.get_param('~env', '')
    vel = rospy.get_param('~vel', '0')
    urdf = rospy.get_param('~urdf', '')

    

    s = str(pathlib.Path(__file__).parent.absolute())
    data_path = s + '/../data/' + base_folder

    rospy.loginfo(data_path)

    # If the path does not exist, create it
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        data_path2 = data_path + '/IMG'
        os.makedirs(data_path2)
    else:
        rospy.logerr('Folder already exists, please try again with a different folder!')
        os._exit(os.EX_OK)

    # yaml file
    imgsize_list = [image_width, image_height]
    string_ints = [str(int) for int in imgsize_list]
    imgsize_str = ",".join(string_ints)

    info_data = dict(

        dataset = dict(
            developer = os.getenv('automec_developer'),
            image_size = imgsize_str,
            frequency = rate_hz,
            linear_velocity = vel,
            environment = env   
        )
    )

    if env == "gazebo":
        info_data["dataset"]["urdf"] = urdf
    else:
        info_data["dataset"]["cam_height"] = cam_height
        info_data["dataset"]["cam_angle"] = cam_angle


    with open(data_path+'/info.yaml', 'w') as outfile:
        yaml.dump(info_data, outfile, default_flow_style=False)

    # Subscribe topics
    # If we have a bool topic, we are recording the linear variable as the boolean.
    # If not, we are recording the linear velocity from the twist
    if vel_cmd_topic != "":
        # Define angular as 0 to prevent errors when we give velocity first instead of angle
        angular = 0
        rospy.Subscriber(twist_cmd_topic, Twist, messageRealReceivedCallback)
        rospy.Subscriber(vel_cmd_topic, Bool, boolReceivedCallback)
    else:
        rospy.Subscriber(twist_cmd_topic, Twist, messageReceivedCallback)

    rospy.Subscriber(image_raw_topic, Image, message_RGB_ReceivedCallback)

    # Create an object of the CvBridge class
    bridge = CvBridge()

    # Create pandas dataframe
    driving_log = pd.DataFrame(columns=['Center', 'Steering', 'Velocity'])

    # set loop rate 
    rate = rospy.Rate(rate_hz)

    # set handler on termination
    signal.signal(signal.SIGINT, signal_handler)

    # only to display saved image counter
    counter = 0

    while not rospy.is_shutdown():

        if not begin_img:
            continue

        cv2.imshow('Robot View', img_rbg)
        cv2.waitKey(1)

        if not begin_cmd:
            continue

        if linear == 0:
            continue

        curr_time = datetime.datetime.now()
        image_name = str(curr_time.year) + '_' + str(curr_time.month) + '_' + str(curr_time.day) + '__' + str(
            curr_time.hour) + '_' + str(curr_time.minute) + '_' + str(curr_time.second) + '__' + str(
            curr_time.microsecond) + str('.jpg')
        # add image, angle and velocity to the driving_log pandas
        row = pd.DataFrame([[image_name, angular, linear]], columns=['Center', 'Steering', 'Velocity'])
        driving_log = driving_log.append(row, ignore_index=True)

        # save image
        dim = (image_width, image_height)
        img_rbg_resize = cv2.resize(img_rbg, dim, interpolation=cv2.INTER_AREA)
        image_saved = Image_pil.fromarray(img_rbg_resize)
        image_saved.save(data_path + '/IMG/' + image_name)
        counter += 1
        rospy.loginfo('Image Saved: %s', counter)
        rate.sleep()


if __name__ == '__main__':
    main()
