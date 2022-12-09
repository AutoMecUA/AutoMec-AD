#!/usr/bin/env python3

import rospy
import cv2

from functools import partial

from std_msgs.msg import String, Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class image:

    def __init__(self,image_args):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/bottom_front_camera/rgb/image_color",Image,partial(self.subscriberCallback,image_args = image_args))

    def subscriberCallback(self,image_msg,image_args = {}):

        image_args["cv_image"] = self.bridge.imgmsg_to_cv2(image_msg,"bgr8") # Passthrough means wtv the encoding was before, it will be retained
        rospy.loginfo("Received image message, image shape is " + str(image_args["cv_image"].shape))

    

def main():
    # -----------------------------
    # Initialization
    # -----------------------------
    rospy.init_node("foot_recognition", anonymous=False)
    cv2.namedWindow("CV_image")

    image_args = dict(cv_image = 0) # Can't be none or imshow will yield error
    bottom_front_camera = image(image_args)


    # -----------------------------
    # Processing
    # -----------------------------
    while(1):
        # print(image_args["cv_image"])
        cv2.imshow("CV_image",image_args["cv_image"])
        cv2.waitKey(1)    
        pass
    # -----------------------------
    # Termination
    # -----------------------------


if __name__ == "__main__":
    main()
