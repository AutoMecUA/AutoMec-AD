#!/usr/bin/env python3
"""
Crosswalk detection
"""

from functools import partial
from typing import Any

import rospy
from cv_bridge.core import CvBridge
from std_msgs.msg import Float32
from sensor_msgs.msg._Image import Image

import stateless_lib
# Callback function to receive image
def message_RGB_ReceivedCallback(message, config: dict):
    
    config['img_rgb'] = config['bridge'].imgmsg_to_cv2(message, "bgr8")

    config['begin_img'] = True


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
    crosswalk_sureness_topic = rospy.get_param('~crosswalk_sureness_topic', '/crosswalk_sureness')
    
    # Create an object of the CvBridge class
    config['bridge'] = CvBridge()

    # Define publisher
    crosswalk_sureness_pub = rospy.Publisher(crosswalk_sureness_topic, Float32, queue_size=10)

    # Subscribe topics
    message_RGB_ReceivedCallback_part = partial(message_RGB_ReceivedCallback, config=config)
    rospy.Subscriber(image_raw_topic, Image, message_RGB_ReceivedCallback_part)
 
    while not rospy.is_shutdown():
        if not config['begin_img']:
            continue
        # Detect crosswalk and publish the content
        crosswalk_sureness_msg = stateless_lib.basic_sureness(config['img_rgb'])
        crosswalk_sureness_pub.publish(crosswalk_sureness_msg)

 
if __name__ == '__main__':
    main()
