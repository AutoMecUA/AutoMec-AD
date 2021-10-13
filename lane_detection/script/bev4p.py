#!/usr/bin/env python3

# bird's eye view calculated from a set of 4 points 
# coordinates in source and destination imagem in order to
# compute perspective transform matrix by opencv method 
# using it in opencv WrapPerspective method , 
# and image undistort
# 
#
# hint: to get the four points coordinates
# use the perspective_explorer node
#
# v1.0.1 by inaciose
#

# Imports
import cv2
import rospy
from sensor_msgs.msg._Image import Image
from cv_bridge.core import CvBridge
import numpy as np

from lib.bev4points import bev

# Callback function to receive image
def message_RGB_ReceivedCallback(message):
    
    # get image from image
    img_rbg = bridge.imgmsg_to_cv2(message, "bgr8")

    # undistorted_image
    undistorted_image = bevo.get_remaped_image(img_rbg)

    # apply transform
    img_ipm = bevo.getWarpPerspective(undistorted_image)

    # debug
    if view_image:
        #cv2.imshow('img', img_rbg)
        cv2.imshow('bev', img_ipm)
        cv2.waitKey(1)

    # Change the cv2 image to imgmsg and publish
    msg_frame = bridge.cv2_to_imgmsg(img_ipm, "bgr8") 
    imagePub.publish(msg_frame)

def main():
    # Global variables
    global bridge
    global imagePub
    global view_image
    global bevo


    # Init Node
    rospy.init_node('image_crop', anonymous=False)

    image_raw_topic = rospy.get_param('~image_raw_topic', '/ackermann_vehicle/camera/rgb/image_raw')
    image_bev_topic = rospy.get_param('~image_bev_topic', '/bev')
    view_image = rospy.get_param('~view', 1)
    rate_hz = rospy.get_param('~rate', 30)

    # image size
    cfg_img = { 'sw': 640,
                'sh': 480}

    # camera distortion
    cfg_d = {   'k1': 0,
                'k2': 0,
                'p1': 0,
                'p2': 0,
                'k3': 0}

    # camera intrinsic
    cfg_k = {   'fx': 563.62,
                'fy': 563.62,
                'sk': 0,                        
                'cx': 340.5,
                'cy': 240.5}

    # points to calculate transformation
    src_points = np.array([[196, 217], [441, 217], [517, 423], [120, 423]], dtype=np.float32)
    dst_points = np.array([[212,202], [427,202], [427,417], [212,417]], dtype=np.float32)

    # create bird eyes view transformation object
    bevo = bev(cfg_img, cfg_d, cfg_k, src_points, dst_points)

    # Subscribe and pubblish topics
    rospy.Subscriber(image_raw_topic, Image, message_RGB_ReceivedCallback)
    imagePub = rospy.Publisher(image_bev_topic, Image, queue_size=2)

    # Create an object of the CvBridge class
    bridge = CvBridge()
    # set loop rate 
    rate = rospy.Rate(rate_hz)

    while  True:    
        rate.sleep()

    rospy.loginfo('Done. exit now!')    

if __name__ == '__main__':
    main()
