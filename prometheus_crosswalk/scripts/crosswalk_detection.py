"""
Crosswalk detection
"""

from functools import partial
import time
from typing import Union, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rospy
from cv_bridge.core import CvBridge
from std_msgs.msg import Bool
from sensor_msgs.msg._Image import Image

import stateless_lib

WHITE = (255, 255, 255)

# Toggle debug mode - features random frame skip
TOGGLE_GRAPH: bool = True  # Show a graph at the end of the run
DEBUG_MODE: bool = False

KILO = 1024
MEGA = 2**20

# Callback function to receive image
def message_RGB_ReceivedCallback(message, config: dict):
    
    config['img_rgb'] = config['bridge'].imgmsg_to_cv2(message, "bgr8")

    config['begin_img'] = True


def isCrosswalk(img: np.ndarray) -> bool:
    """Verifies if the image qualifies as having a crossroad in it

    Assumes the image regards to photo of a road, either real or virtual (e.g.: gazebo)

    :param img: Matrix representation of the image
    """
    sureness_threshold: float = .5

    return stateless_lib.basic_sureness(img) > sureness_threshold


def main():
    # Global variables
    config: dict[str, Any] = dict(
        bridge=None,
        img_rgb=None,
        begin_img=False,
    )

    # Init Node
    rospy.init_node('crosswalk_detection', anonymous=False)

    # Retrieving parameters
    image_raw_topic = rospy.get_param('~image_raw_topic', '/top_front_camera/rgb/image_raw')
    crosswalk_detected_topic = rospy.get_param('~crosswalk_detected_topic', '/crosswalk_detected')
    
    # Create an object of the CvBridge class
    config['bridge'] = CvBridge()

    # Define publisher
    crosswalk_detected_pub = rospy.Publisher(crosswalk_detected_topic, Bool, queue_size=10)

    # Subscribe topics
    message_RGB_ReceivedCallback_part = partial(message_RGB_ReceivedCallback, config=config)
    rospy.Subscriber(image_raw_topic, Image, message_RGB_ReceivedCallback_part)
 
    while not rospy.is_shutdown():
        if not config['begin_img']:
            continue

        crosswalk_detected_msg = isCrosswalk(config['img_rgb'])

        crosswalk_detected_pub.publish(crosswalk_detected_msg)

 
if __name__ == '__main__':
    main()
