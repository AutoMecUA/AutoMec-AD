#!/usr/bin/env python3

# Imports
import os
import pathlib
import shutil
import socket
from datetime import datetime
from functools import partial
from typing import Any

import cv2
import pandas as pd
import rospy
import yaml
from PIL import Image as Image_pil
from cv_bridge.core import CvBridge
from geometry_msgs.msg._Twist import Twist
from sensor_msgs.msg._Image import Image
from std_msgs.msg import Bool


# Calback Function to receive the cmd values
def messageReceivedCallback(message, config: dict):

    config['angular'] = float(message.angular.z)
    config['linear'] = float(message.linear.x)

    config['begin_cmd'] = True


def messageRealReceivedCallback(message, config: dict):

    config['angular'] = float(message.angular.z)


def boolReceivedCallback(message, config: dict):

    if message.data:
        config['linear'] = 1
        config['begin_cmd'] = True
    else:
        config['linear'] = 0
        config['begin_cmd'] = False


# Callback function to receive image
def message_RGB_ReceivedCallback(message, config: dict):

    config['img_rbg'] = config['bridge'].imgmsg_to_cv2(message, "bgr8")

    config['begin_img'] = True


def save_dataset(date, info_data, data_path, driving_log):
    rospy.loginfo('EXITING...')
    driving_log.to_csv(data_path + '/driving_log.csv', mode='a', index=False, header=False)
    info_data['dataset']['image_number'] = len(list(os.listdir(data_path + "/IMG/")))

    info_data['dataset']['date'] = date + " until " + datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    comments = input("[info.yaml] Additional comments about the dataset: ")
    info_data['dataset']['comments'] = comments
    with open(data_path+'/info.yaml', 'w') as outfile:
        yaml.dump(info_data, outfile, default_flow_style=False)
    rospy.signal_shutdown("All done, exiting ROS...")


def main():
    # Global variables
    config: dict[str, Any] = dict(
        angular=None,
        linear=None,
        bridge=None,
        img_rbg=None,
        begin_img=False,
        begin_cmd=False,
    )

    # Init Node
    rospy.init_node('write_data', anonymous=False)

    image_raw_topic = rospy.get_param('~image_raw_topic', '/ackermann_vehicle/camera/rgb/image_raw')
    twist_cmd_topic = rospy.get_param('~twist_cmd_topic', '/cmd_vel')
    vel_cmd_topic = rospy.get_param('~vel_cmd_topic', '')
    rate_hz = rospy.get_param('~rate', 30)
    image_width = rospy.get_param('~width', 320)
    image_height = rospy.get_param('~height', 160)

    # params only used in yaml file
    cam_pose = rospy.get_param('~cam_pose', '')
    env = rospy.get_param('~env', '')
    vel = rospy.get_param('~vel', '0')
    urdf = rospy.get_param('~urdf', '')
    challenge = "driving"  # rospy.get_param('~challenge', 'driving') # TODO: add this to launch files...

    s = str(pathlib.Path(__file__).parent.absolute())
    date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    data_path = f'{s}/../data/' + env + "-" + datetime.now().strftime("%d-%m-%Hh%Mm%Ss")

    rospy.loginfo(data_path)

    # If the path does not exist, create it
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        data_path_imgs = data_path + '/IMG'
        os.makedirs(data_path_imgs)
    else:
        rospy.logerr('Folder already exists, please try again with a different folder!')
        exit(1)

    # yaml file
    imgsize_list = [image_width, image_height]
    string_ints = [str(size) for size in imgsize_list]
    imgsize_str = ",".join(string_ints)

    info_data = dict(
        dataset=dict(
            developer=os.getenv('automec_developer') or socket.gethostname(),
            cam_pose=cam_pose if env != 'gazebo' else urdf,
            environment=env,
            frequency=rate_hz,
            image_size=imgsize_str,
            image_number=0,
            linear_velocity=vel,
            challenge=challenge
        )
    )

    # Subscribe topics
    # If we have a bool topic, we are recording the linear variable as the boolean.
    # If not, we are recording the linear velocity from the twist
    if vel_cmd_topic != "":
        # Define angular as 0 to prevent errors when we give velocity first instead of angle
        config['angular'] = 0
        messageRealReceivedCallback_part = partial(messageRealReceivedCallback, config)
        boolReceivedCallback_part = partial(boolReceivedCallback, config=config)

        rospy.Subscriber(twist_cmd_topic, Twist, messageRealReceivedCallback_part)
        rospy.Subscriber(vel_cmd_topic, Bool, boolReceivedCallback_part)
    else:
        messageReceivedCallback_part = partial(messageReceivedCallback, config=config)
        rospy.Subscriber(twist_cmd_topic, Twist, messageReceivedCallback_part)

    message_RGB_ReceivedCallback_part = partial(message_RGB_ReceivedCallback, config=config)
    rospy.Subscriber(image_raw_topic, Image, message_RGB_ReceivedCallback_part)

    # Create an object of the CvBridge class
    config['bridge'] = CvBridge()

    # Create pandas dataframe
    driving_log = pd.DataFrame(columns=['Center', 'Steering', 'Velocity'])

    # set loop rate
    rate = rospy.Rate(rate_hz)

    # only to display saved image counter
    counter = 0

    # read opencv key
    key = -1

    while True:
        if not config['begin_img']:
            continue

        cv2.imshow('Robot View', config['img_rbg'])
        key = cv2.waitKey(1)

        if not config['begin_cmd']:
            continue

        if config['linear'] == 0:
            continue

        curr_time = datetime.now()
        image_name = f'{str(curr_time.year)}_{str(curr_time.month)}_{str(curr_time.day)}_' \
                     f'_{str(curr_time.hour)}_{str(curr_time.minute)}_{str(curr_time.second)}_' \
                     f'_{str(curr_time.microsecond)}.jpg'

        # add image, angle and velocity to the driving_log pandas
        row = pd.DataFrame(data=[[image_name, config['angular'], config['linear']]],
                           columns=['Center', 'Steering', 'Velocity'])
        driving_log = driving_log.append(row, ignore_index=True)

        # save image
        dim = (image_width, image_height)
        img_rbg_resize = cv2.resize(config['img_rbg'], dim, interpolation=cv2.INTER_AREA)
        image_saved = Image_pil.fromarray(img_rbg_resize)
        image_saved.save(data_path + '/IMG/' + image_name)
        counter += 1
        rospy.loginfo('Image Saved: %s', counter)
        rate.sleep()

    # save on shutdown...
    if key == ord('s'):  
        save_dataset(date, info_data, data_path, driving_log)
        exit(0)

    if key == ord('q'):
        confirmation = input("\n\nYou have pressed q[uit]: are you sure you want to close"
                             " WITHOUT saving the dataset? (type 'yes' TO DISCARD the dataset,"
                             " type 'no' or 'save' to SAVE the dataset): ")
        if confirmation == "yes":
            shutil.rmtree(data_path)
            rospy.signal_shutdown("All done, exiting ROS...")
        elif confirmation in {'no', 'save'}:
            save_dataset(date, info_data, data_path, driving_log)
            exit(0)


if __name__ == '__main__':
    main()
