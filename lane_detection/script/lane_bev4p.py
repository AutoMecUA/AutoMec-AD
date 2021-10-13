#!/usr/bin/env python3

#
# hint: to get the four points coordinates
# use the perspective_explorer node
#
# v0.1.0 by inaciose
#

# Imports
import cv2
import rospy
from sensor_msgs.msg._Image import Image
from cv_bridge.core import CvBridge
import numpy as np

from lib.bev4points import bev
from lib.simplelane import simplelane

# Callback function to receive image
def message_RGB_ReceivedCallback(message):
    # libs: bev (bevo), lane (laneo)
    
    # get image from image
    img_rbg = bridge.imgmsg_to_cv2(message, "bgr8")
    #img = img_rbg

    # undistorted_image
    uimg = bevo.get_remaped_image(img_rbg)

    # apply bev transform
    bev_img = bevo.getWarpPerspective(uimg)

    line_image = laneo.image_pipeline(bev_img)


    # debug
    if view_image:
        #cv2.imshow('img', img_rbg)
        cv2.imshow('final', line_image)
        cv2.waitKey(1)

    #final_img = cv2.cvtColor(line_image, cv2.COLOR_GRAY2RGB)


    final_img = line_image

    # Change the cv2 image to imgmsg and publish
    #msg_frame = bridge.cv2_to_imgmsg(cbev_canimg) 
    msg_frame = bridge.cv2_to_imgmsg(final_img, "bgr8") 
    imagePub.publish(msg_frame)

def main():
    # Global variables
    global bridge
    global imagePub
    global view_image
    global bevo
    global laneo


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

    # points to calculate transformation to bev
    src_points = np.array([[196, 217], [441, 217], [517, 423], [120, 423]], dtype=np.float32)
    dst_points = np.array([[212,202], [427,202], [427,417], [212,417]], dtype=np.float32)
    # create bird eyes view transformation object
    bevo = bev(cfg_img, cfg_d, cfg_k, src_points, dst_points)

    # define region of interest
    roi_points = np.array([[10, 0], [629, 0], [629, 479], [10, 479]], dtype=np.int32)
    # create lane trancking object
    laneo = simplelane(cfg_img, roi_points)


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
