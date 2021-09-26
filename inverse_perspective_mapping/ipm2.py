#!/usr/bin/env python3

#
# based on: https://stackoverflow.com/questions/57439977/transforming-perspective-view-to-a-top-view
# issue: how to select the 4 source points coordinates and the the corresponent destination coordinates?
#

# Imports
import cv2
import rospy
from sensor_msgs.msg._Image import Image
from cv_bridge.core import CvBridge

import numpy as np

# Callback function to receive image
def message_RGB_ReceivedCallback(message):
    
    # get image from image
    img_rbg = bridge.imgmsg_to_cv2(message, "bgr8")

    # debug
    for pt in pts:
        cv2.circle(img_rbg, tuple(pt.astype(np.int)), 4, (0,255,0), -1)

    # apply transform
    img_ipm = cv2.warpPerspective(img_rbg, ipm_matrix, img_rbg.shape[:2][::-1])

    # debug
    cv2.imshow('img', img_rbg)
    cv2.imshow('ipm', img_ipm)
    cv2.waitKey(1)

    # Change the cv2 image to imgmsg and publish
    msg_frame = bridge.cv2_to_imgmsg(img_ipm, "bgr8") 
    imagePub.publish(msg_frame)

def main():
    # Global variables
    global bridge
    global imagePub
    global pts
    global ipm_matrix

    # Init Node
    rospy.init_node('image_crop', anonymous=False)

    image_raw_topic = rospy.get_param('~image_raw_topic', '/ackermann_vehicle/camera/rgb/image_raw')
    image_crop_topic = rospy.get_param('~image_raw_topic', '/bev') 
   
    rate_hz = rospy.get_param('~rate', 30)

    # Subscribe and pubblish topics
    rospy.Subscriber(image_raw_topic, Image, message_RGB_ReceivedCallback)
    imagePub = rospy.Publisher(image_crop_topic, Image, queue_size=2)

    # Create an object of the CvBridge class
    bridge = CvBridge()

    # ipm points & matrix
    pts = np.array([[86, 174], [447, 174], [481, 282], [2, 282]], dtype=np.float32)
    ipm_pts = np.array([[177,193], [320,193], [320,254], [177,254]], dtype=np.float32)

    ipm_matrix = cv2.getPerspectiveTransform(pts, ipm_pts)

    # set loop rate 
    rate = rospy.Rate(rate_hz)

    while  True:    
        rate.sleep()

    rospy.loginfo('Done. exit now!')    

if __name__ == '__main__':
    main()
