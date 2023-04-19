#!/usr/bin/env python3

# Imports
from functools import partial
from typing import Any

import cv2
import numpy as np
import rospy
import yaml
from sensor_msgs.msg._Image import Image
from std_msgs.msg import Float32 , String
from cv_bridge.core import CvBridge
import torch
from torchvision import transforms
import pathlib
import os
import datetime
import matplotlib.pyplot as plt


# Custom imports
from models.yolo import Model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

def LoadModel(model_path,model,device):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device) # move the model variable to the gpu if one exists
    return model

def imgRgbCallback(message, config):

    config['img_rgb'] = config['bridge'].imgmsg_to_cv2(message, "rgb8")

    config["begin_img"] = True

def modelSteeringCallback(message, config):
    model_name = message.data
    automec_path = os.environ.get('AUTOMEC_DATASETS')
    path = f'{automec_path}/models/{model_name}/{model_name}.pt'
   
    rospy.loginfo('Using model: %s', path)
    device = f'cuda:0' if torch.cuda.is_available() else 'cpu' # cuda: 0 index of gpu
    model = eval(info_loaded['model']['ml_arch']['name'])
    model = LoadModel(path,model,device)
    model.eval()
    config['model'] = model


def main():
    config: dict[str, Any] = dict(vel=None, img_rgb=None,
                                  bridge=None, begin_img=None)
    # Defining starting values
    config["begin_img"] = False
    config["vel"] = 0
    config["bridge"] = CvBridge()

    # Init Node
    rospy.init_node('ml_signal_recognition', anonymous=False)

    # Getting parameters
    image_raw_topic = rospy.get_param('~image_raw_topic', '/top_right_camera/image_raw')
    #model_steering_topic = rospy.get_param('~model_steering_topic', '/model_steering')
    model_name = rospy.get_param('/model_name', '')

    # Defining path to model
    automec_path = os.environ.get('AUTOMEC_DATASETS')
    path = f'{automec_path}/models/{model_name}/{model_name}.pkl'

    # Retrieving info from yaml
    with open(f'{automec_path}/models/{model_name}/{model_name}.yaml') as file:
        info_loaded = yaml.load(file, Loader=yaml.FullLoader)

    rospy.loginfo('Using model: %s', path)
    device = f'cuda:0' if torch.cuda.is_available() else 'cpu' # cuda: 0 index of gpu
    config['model'] = eval(info_loaded['model']['ml_arch']['name'])
    config['model'] = LoadModel(path,config['model'],device)
    config['model'].eval()

    PIL_to_Tensor = transforms.Compose([
                    transforms.ToTensor()
                    ])

    # Partials
    imgRgbCallback_part = partial(imgRgbCallback, config=config)

    changeModelCallback = partial(modelSteeringCallback, config=config)

    # Subscribe and publish topics
    rospy.Subscriber(image_raw_topic, Image, imgRgbCallback_part)
    rospy.Subscriber('/set_model', String, changeModelCallback)
    #model_steering_pub = rospy.Publisher(model_steering_topic, Float32, queue_size=10)

    # Frames per second
    rate = rospy.Rate(30)

    while not rospy.is_shutdown():

        if config["begin_img"] is False:
            continue

        # Predict angle
        image = np.array(config["img_rgb"])
        image = image[0,:,:,:]
        image = PIL_to_Tensor(image)
        image = image.unsqueeze(0)
        image = image.to(device, dtype=torch.float)
        label_t_predicted = config['model'].forward(image)
        steering = float(label_t_predicted)
        # Publish angle
        model_steering_pub.publish(steering)
        rate.sleep()


if __name__ == '__main__':
    main()
