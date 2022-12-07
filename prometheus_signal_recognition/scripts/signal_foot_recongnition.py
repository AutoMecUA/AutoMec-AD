#!/usr/bin/env python3

import rospy
import cv2

from functools import partial

from std_msgs.msg import String, Header
from sensor_msgs.msg import Image


def main():
    # -----------------------------
    # Initialization
    # -----------------------------
    rospy.init_node("foot_recognition", anonymous=False)

    # -----------------------------
    # Processing
    # -----------------------------

    rospy.Subscriber("/bottom_front_camera/rgb/image_color",Image,)


    # -----------------------------
    # Termination
    # -----------------------------
if __name__ == "__main__":
    main()
