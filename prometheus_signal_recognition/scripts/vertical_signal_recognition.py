#!/usr/bin/env python3

"""
   This script receives the image from the /top_right_camera/image_raw camera via ROS
"""

# Imports
from functools import partial
from typing import Any

import cv2
import rospy
from cv_bridge.core import CvBridge
from sensor_msgs.msg import Image

# ______________________________________________________________
# Imports
import argparse
import os
import sys
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory

# My imports
from src.models.common import DetectMultiBackend
from src.utils_yolo.general import (LOGGER, check_file, check_img_size, check_imshow, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from src.utils_yolo.utils_yolo import (IMG_FORMATS, LoadImages, Annotator, colors, save_one_box,
                           select_device, time_sync)
# ______________________________________________________________


def imgRgbCallback(message, config):
    """Callback for changing the image.
    Args:
        message (Image): ROS Image message.
        config (dict): Dictionary with the configuration. 
    """

    config['img_rgb'] = config['bridge'].imgmsg_to_cv2(message, "bgr8")

    #config['img_rgb'] = cv2.cvtColor(config['img_rgb'], cv2.COLOR_BGR2RGB)

    config["begin_img"] = True


def main():
    ############################
    # Initialization           #
    ############################
    # Defining starting values
    config: dict[str, Any] = dict(img_rgb=None, bridge=None, begin_img=False)

    config["begin_img"] = False
    config["bridge"] = CvBridge()


    # Init Node
    rospy.init_node('vertical_signal_recognition', anonymous=False)

    # Getting parameters
    image_raw_topic = rospy.get_param('~image_raw_topic', '/top_right_camera/image_raw')

    imgRgbCallback_part = partial(imgRgbCallback, config = config)

    # Subscribe and publish topics
    rospy.Subscriber(image_raw_topic, Image, imgRgbCallback_part)
 
    #model_steering_pub = rospy.Publisher(model_steering_topic, Float32, queue_size=10)

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
        image = config['img_rgb']
        
        
        # show image      
        cv2.imshow('image',image)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            rospy.loginfo('Leter "q" pressed, exiting the program')
            cv2.destroyAllWindows()
            rospy.signal_shutdown("Manual shutdown")

        rate.sleep()


if __name__ == '__main__':
    main()