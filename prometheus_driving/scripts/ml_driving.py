#!/usr/bin/env python3

# Imports
from functools import partial
from typing import Any

import cv2
import numpy as np
import rospy
import yaml
from sensor_msgs.msg._Image import Image
from std_msgs.msg import String, Float32
from cv_bridge.core import CvBridge
from tensorflow.keras.models import load_model
import pathlib

def preProcess(img):
    # Define Region of interest
    #img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = img[40:, :]  #cut the 40 first lines
    img = cv2.resize(img, (320, 160))
    img = img/255
    return img



def imgRgbCallback(message, config):

    config['img_rgb'] = config['bridge'].imgmsg_to_cv2(message, "bgr8")

    config["begin_img"] = True


def main():
    config: dict[str, Any] = dict(vel=None, img_rgb=None,
                                  bridge=None, begin_img=None)
    # Defining starting values
    config["begin_img"] = False
    config["vel"] = 0
    config["bridge"] = CvBridge()
    # Define cv2 windows
    win_name = 'Robot View'
    cv2.namedWindow(winname=win_name)

    # Init Node
    rospy.init_node('ml_driving', anonymous=False)

    # Getting parameters
    image_raw_topic = rospy.get_param('~image_raw_topic', '/top_front_camera/rgb/image_raw')
    model_steering_topic = rospy.get_param('~model_steering_topic', '/model_steering')
    model_name = rospy.get_param('/model_name', '')

    # Defining path to model
    s = str(pathlib.Path(__file__).parent.absolute())
    path = f'{s}/../models/{model_name}.h5'

    # Retrieving info from yaml
    with open(f'{s}/../models/{model_name}.yaml') as file:
        info_loaded = yaml.load(file, Loader=yaml.FullLoader)

    rospy.loginfo('Using model: %s', path)
    model = load_model(path)

    # Partials
    imgRgbCallback_part = partial(imgRgbCallback, config=config)

    # Subscribe and publish topics
    rospy.Subscriber(image_raw_topic, Image, imgRgbCallback_part)
    model_steering_pub = rospy.Publisher(model_steering_topic, Float32, queue_size=10)

    # Frames per second
    rate = rospy.Rate(30)

    while not rospy.is_shutdown():

        if config["begin_img"] is False:
            continue

        resized_img = preProcess(config["img_rgb"])

        # Predict angle
        image = np.array([resized_img])
        steering = float(model.predict(image))
        model_steering_pub.publish(steering)

        rate.sleep()


if __name__ == '__main__':
    main()
