#!/usr/bin/env python3

import rospy
import cv2

from functools import partial

from std_msgs.msg import String, Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def subscriberCallback(image_msg,rgb_args):

    rgb_args["cv_image"] = rgb_args['bridge'].imgmsg_to_cv2(image_msg,"bgr8") # Passthrough means wtv the encoding was before, it will be retained

    rospy.loginfo("Received image message, image shape is " + str(rgb_args["cv_image"].shape))

    

def main():
    # -----------------------------
    # Initialization
    # -----------------------------
    rospy.init_node("foot_recognition", anonymous=False)

    rgb_args = dict(bridge = CvBridge(),cv_image = None)

    rospy.Subscriber("/bottom_front_camera/rgb/image_color",Image,partial(subscriberCallback,rgb_args = rgb_args))

    cv2.namedWindow("CV_image")
    # -----------------------------
    # Processing
    # -----------------------------
    while(1):
        cv2.imshow("CV_image",rgb_args["cv_image"])
        cv2.waitKey(1)    

    # -----------------------------
    # Termination
    # -----------------------------


if __name__ == "__main__":
    main()
