#!/usr/bin/env python3

import rospy
import cv2

from functools import partial

from std_msgs.msg import String, Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class image:

    #* Class variables-----

    #* ---------------------
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/bottom_front_camera/rgb/image_color",Image,self.subscriberCallback)
        self.image_args = {} # Contains cv_image
        self.begin_image = False

    def subscriberCallback(self,image_msg):

        self.begin_image = True # After receiving first image will be forever true

        #! If I assign directly to cv_image key, it will show flickers of the non flipped image 
        self.image_args["processing_image"] = self.bridge.imgmsg_to_cv2(image_msg,"bgr8") # Passthrough means wtv the encoding was before, it will be retained
        self.flipImage('x','y')
        rospy.loginfo("Received image message, image shape is " + str(self.image_args["cv_image"].shape))

    def showImage(self):

        if self.begin_image == False: # Only shows after first image is processed
            return

        cv2.namedWindow("CV_image")
        cv2.imshow("CV_image",self.image_args["cv_image"])
        pressed_key = cv2.waitKey(1) & 0xFF # To prevent NumLock issue

        if pressed_key == ord('q'):
            print("Quitting program")
            cv2.destroyAllWindows
            rospy.signal_shutdown("Order to quit") # Stops ros
            exit()                                 # Exits python script
    
    def flipImage(self,*args):
        
        for arg in args:
            try:
                if arg == 'x':
                    self.image_args["processing_image"] = cv2.flip(self.image_args["processing_image"],1)
                elif arg == 'y':
                    self.image_args["processing_image"] = cv2.flip(self.image_args["processing_image"],0)
            except :
                print("Unallowed arguments, should be 'x' or 'y' ")

        self.image_args["cv_image"] = self.image_args["processing_image"] # Should find better way to do this, if I change cv image in the loop it will flicker 

def main():
    # -----------------------------
    # Initialization
    # -----------------------------
    rospy.init_node("foot_recognition", anonymous=False)
    bottom_front_camera = image()

    # -----------------------------
    # Processing
    # -----------------------------
    while(1):
        bottom_front_camera.showImage()
    # -----------------------------
    # Termination
    # -----------------------------


if __name__ == "__main__":
    main()
