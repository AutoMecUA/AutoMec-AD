#!/usr/bin/env python3

# Imports
from functools import partial
from typing import Any

import cv2
import numpy as np
import rospy
import yaml
from geometry_msgs.msg._Twist import Twist
from sensor_msgs.msg._Image import Image
from std_msgs.msg import String
from cv_bridge.core import CvBridge
import torch
from torchvision import transforms
import pathlib
import os

from models.cnn_nvidia import Nvidia_Model
from models.cnn_rota import Rota_Model
from src.utils import LoadModel


def preProcess(img):
    # Define Region of interest
    #img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    #img = img[40:, :]  #cut the 40 first lines
    img = cv2.resize(img, (320, 160))
    img = img/255
    return img


def message_RGB_ReceivedCallback(message, config):

    config['img_rgb'] = config['bridge'].imgmsg_to_cv2(message, "passthrough")

    config["begin_img"] = True


def signalCallback(message, config):
    config['signal'] = message.data


def main():
    config: dict[str, Any] = dict(vel=None, signal=None, img_rgb=None,
                                  bridge=None, begin_img=None, twist_linear_x=None)

    # Defining starting values
    config["begin_img"] = False
    config["vel"] = 0
    config["bridge"] = CvBridge()
    twist = Twist()

    # Init Node
    rospy.init_node('ml_driving', anonymous=False)

    # Getting parameters
    image_raw_topic = rospy.get_param('~image_raw_topic', '/top_front_camera/rgb/image_raw')
    twist_cmd_topic = rospy.get_param('~twist_cmd_topic', '')
    signal_cmd_topic = rospy.get_param('~signal_cmd_topic', '')
    model_name = rospy.get_param('~model_name', '')

    if model_name == "":
        model_name = input('Please define the name of the model to be used: ')
    # Defining path to model
    s = str(pathlib.Path(__file__).parent.absolute())
    automec_path = os.environ.get('AUTOMEC_DATASETS')
    path = f'{automec_path}/models/{model_name}.pkl'

    # Retrieving info from yaml
    with open(f'{automec_path}/models/{model_name}.yaml') as file:
        info_loaded = yaml.load(file, Loader=yaml.FullLoader)
        linear_velocity = info_loaded['dataset']['linear_velocity'] 
    
    rospy.loginfo('Using model: %s', path)
    device = f'cuda:0' if torch.cuda.is_available() else 'cpu' # cuda: 0 index of gpu
    model = Rota_Model()
    model= LoadModel(path,model,device)
    model.eval()

    PIL_to_Tensor = transforms.Compose([
                    transforms.ToTensor()
                    ])

    # Partials
    message_RGB_ReceivedCallback_part = partial(message_RGB_ReceivedCallback, config=config)
    signalCallback_part = partial(signalCallback, config=config)

    # Subscribe and publish topics
    rospy.Subscriber(image_raw_topic, Image, message_RGB_ReceivedCallback_part)
    rospy.Subscriber(signal_cmd_topic, String, signalCallback_part)
    twist_pub = rospy.Publisher(twist_cmd_topic, Twist, queue_size=10)

    # Frames per second
    rate = rospy.Rate(30)

    while not rospy.is_shutdown():

        if config["begin_img"] is False:
            continue

        resized_img = preProcess(config["img_rgb"])

        cv2.imshow('Robot View Processed', resized_img)
        cv2.imshow('Robot View', config["img_rgb"])
        key = cv2.waitKey(1)

        # Predict angle
        image = np.array([resized_img])
        image = image[0,:,:,:]
        image = PIL_to_Tensor(image)
        image = image.unsqueeze(0)
        image = image.to(device, dtype=torch.float)
        steering = float(model.forward(image))
        angle = steering
        
        # Depending on the message from the callback, choose what to do
        if config['signal'] == 'pForward':
            print('Detected pForward, moving forward')
            config["vel"] = linear_velocity
        elif config['signal'] == 'pStop':
            config["vel"] = 0
            print('Detected pStop, stopping')
        elif config['signal'] == 'pChess':
            config["vel"] = 0
            print('Detected chessboard, stopping the program')
            exit(0)
        else:
            config["vel"] = 0

        # Send twist
        twist.linear.x = config["vel"]
        twist.linear.y = 0
        twist.linear.z = 0
        twist.angular.x = 0
        twist.angular.y = 0
        twist.angular.z = angle

        # Stop the script
        if key == ord('q'):
            twist.linear.x = 0
            twist.angular.z = 0
            twist_pub.publish(twist)
            print('Stopping the autonomous driving')

            # Recording comments
            comments = input("[info.yaml] Additional comments about the model: ")
            if 'driving_comments' not in info_loaded['model'].keys():
                info_loaded['model']['driving_comments'] = comments
            else:
                info_loaded['model']['driving_comments'] = info_loaded['model']['driving_comments'] + '; ' + comments
            
            # Recording comments
            model_eval = input("[info.yaml] Evaluate the model on a scale from 0 (bad) to 10 (good): ") + '/10'
            if 'driving_model_eval' not in info_loaded['model'].keys():
                info_loaded['model']['driving_model_eval'] = model_eval
            else:
                info_loaded['model']['driving_model_eval'] = info_loaded['model']['driving_model_eval'] + '; ' + model_eval
            
            # Saving yaml
            with open(f'{s}/../models/{model_name}.yaml', 'w') as outfile:
                yaml.dump(info_loaded, outfile, default_flow_style=False)
            exit(0)

        # To avoid any errors
        twist_pub.publish(twist)

        rate.sleep()


if __name__ == '__main__':
    main()
