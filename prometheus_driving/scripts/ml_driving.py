#!/usr/bin/env python3

"""
    Script for driving the car using a Deep-learning model.
    The model is loaded from a .pkl file and the image is obtained from a ROS topic.
    The model is loaded from the path specified in the parameter 'model_name'.
    The image is obtained from the topic specified in the parameter 'image_raw_topic'.
    The steering angle is published in the topic specified in the parameter 'model_steering_topic'.
    The model can be changed by publishing the name of the model in the topic '/set_model'.
"""

# Imports
from functools import partial
from typing import Any

import cv2
import rospy
import yaml
from sensor_msgs.msg._Image import Image
from std_msgs.msg import Float32 , String
from geometry_msgs.msg import Twist
from cv_bridge.core import CvBridge
import torch
from torchvision import transforms
import os

# Custom imports
from models.cnn_nvidia import Nvidia_Model
from models.cnn_rota import Rota_Model
from models.mobilenetv2 import MobileNetV2
from models.inceptionV3 import InceptionV3
from models.vgg import MyVGG
from models.resnet import ResNet
from models.lstm import LSTM
from models.resnet_imported import ResNetV1
from models.transformer import MyViT
from src.utils import LoadModel

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

def cmdVelCallback(message,config):
    
    config['vel'] = message.linear.x 

def imgRgbCallback(message, config):
    """Callback for changing the image.
    Args:
        message (Image): ROS Image message.
        config (dict): Dictionary with the configuration. 
    """

    config['img_rgb'] = config['bridge'].imgmsg_to_cv2(message, "passthrough")

    config['img_rgb'] = cv2.cvtColor(config['img_rgb'], cv2.COLOR_BGR2RGB)

    config["begin_img"] = True

def modelSteeringCallback(message, config):
    """Callback for changing the model.
    Args:
        message (String): ROS String message containing the model name.
        config (dict): Dictionary with the configuration.
    """

    # Gets the information necessary to run the model
    model_name = message.data
    automec_path = os.environ.get('AUTOMEC_DATASETS')
    path = f'{automec_path}/models/{model_name}/{model_name}.pkl'
    # Retrieving info from yaml
    with open(f'{automec_path}/models/{model_name}/config.yaml') as file:
        info_loaded = yaml.load(file, Loader=yaml.FullLoader)

    # Loads the model
    rospy.loginfo('Using model: %s', path)
    device = f'cuda:0' if torch.cuda.is_available() else 'cpu' # cuda: 0 index of gpu
    model = eval(info_loaded['model']['ml_arch']['name'])
    model = LoadModel(path,model,device)
    model.eval()
    config['model'] = model
    rgb_mean = info_loaded['model']['rgb_mean']
    rgb_std = info_loaded['model']['rgb_std']
    image_size = info_loaded['model']['image_size']

    # Sets the transforms to preprocess the image
    config['transforms'] = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(rgb_mean, rgb_std)
    ])


def main():
    ############################
    # Initialization           #
    ############################
    # Defining starting values
    config: dict[str, Any] = dict(vel=None, img_rgb=None, bridge=None, begin_img=None)
    config["begin_img"] = False
    config["vel"] = 0
    config["bridge"] = CvBridge()

    # Init Node
    rospy.init_node('ml_driving', anonymous=False)

    # Getting parameters
    image_raw_topic = rospy.get_param('~image_raw_topic', '/top_front_camera/rgb/image_color')
    cmd_vel_topic = rospy.get_param('~cmd_vel_topic', '/cmd_vel_tmp')
    model_steering_topic = rospy.get_param('~model_steering_topic', '/model_steering')
    model_name = rospy.get_param('/model_name', '')

    # Defining path to model
    automec_path = os.environ.get('AUTOMEC_DATASETS')
    path = f'{automec_path}/models/{model_name}/{model_name}.pkl'

    # Retrieving info from yaml
    with open(f'{automec_path}/models/{model_name}/config.yaml') as file:
        info_loaded = yaml.load(file, Loader=yaml.FullLoader)
    # Loads the model
    rospy.loginfo('Using model: %s', path)
    device = f'cuda:0' if torch.cuda.is_available() else 'cpu' # cuda: 0 index of gpu
    config['model'] = eval(info_loaded['model']['ml_arch']['name'])
    config['model'] = LoadModel(path,config['model'],device)
    config['model'].eval()
    rgb_mean = info_loaded['model']['rgb_mean']
    rgb_std = info_loaded['model']['rgb_std']
    image_size = info_loaded['model']['image_size']
    # Sets the transforms to preprocess the image
    config['transforms'] = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize(image_size),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            transforms.Normalize(rgb_mean, rgb_std)
                            ])

    # Partials
    imgRgbCallback_part = partial(imgRgbCallback, config=config)

    changeModelCallback = partial(modelSteeringCallback, config=config)
    
    cmdVelCallback_part  = partial(cmdVelCallback, config=config)

    # Subscribe and publish topics
    rospy.Subscriber(image_raw_topic, Image, imgRgbCallback_part)
    rospy.Subscriber('/set_model', String, changeModelCallback)
    rospy.Subscriber(cmd_vel_topic, Twist, cmdVelCallback_part)
    model_steering_pub = rospy.Publisher(model_steering_topic, Float32, queue_size=10)

    # Frames per second
    rate = rospy.Rate(30)

    ############################
    # Main loop                #
    ############################
    while not rospy.is_shutdown():
        # If there is no image, do nothing
        if config["begin_img"] is False:
            continue

        ############################
        # Predicts the steering    #
        ############################
        image = config['transforms'](config["img_rgb"])
        image = image.unsqueeze(0)
        image = image.to(device, dtype=torch.float)
        label_t_predicted = config['model'].forward(image,config['vel'])
        steering = float(label_t_predicted)
        # Publish angle
        model_steering_pub.publish(steering)

        print(config['vel'])


        rate.sleep()


if __name__ == '__main__':
    main()
