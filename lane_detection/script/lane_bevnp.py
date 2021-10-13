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

from lib.bev import bev
from lib.simplelane import simplelane

# Callback function to receive image
def message_RGB_ReceivedCallback(message):
    
    # get image from image
    img_rbg = bridge.imgmsg_to_cv2(message, "bgr8")

    undistorted_image = bevo.get_remaped_image(img_rbg)
    bev_img = bevo.get_bev_image(undistorted_image)

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
    image_crop_topic = rospy.get_param('~image_raw_topic', '/bev')
    view_image = rospy.get_param('~view', 1)
   
    rate_hz = rospy.get_param('~rate', 30)

    # image size
    cfg_img = { 'sw': 640,
                'sh': 480}


    # radial and tangential distortion coefficients
    cfg_distortion = {   'k1': 0.001593932351222154,
                            'k2': -0.004430239769797794,
                            'p1': 0.0002036043769375994,
                            'p2': -3.141393111217492e-06,
                            'k3': 0}

    # Intrinsic parameters are specific to a camera
    cfg_intrinsic = {    'fx': 564.1794158410564,
                            'fy': 563.8954010066177,
                            'sk': 0,                        
                            'cx': 339.585354927923,
                            'cy': 240.8593643042658,
                            'iw': 640,
                            'ih': 480}

    # Extrinsic parameters corresponds to rotation and translation vectors 
    cfg_extrinsic = {    'tx': 10,
                            'ty': 0,                        
                            'tz': 547,                        
                            'pitch': 34.4,                        
                            'roll': 90,                        
                            'yaw': 90}

    # Subscribe and pubblish topics
    rospy.Subscriber(image_raw_topic, Image, message_RGB_ReceivedCallback)
    imagePub = rospy.Publisher(image_crop_topic, Image, queue_size=2)

    # Create an object of the CvBridge class
    bridge = CvBridge()

    # create ipm class instance
    #bevo = ipm(cfg_distortion, cfg_intrinsic, cfg_extrinsic)
    bevo = bev(cfg_distortion, cfg_intrinsic, cfg_extrinsic)

    # define region of interest
    roi_points = np.array([[10, 0], [629, 0], [629, 479], [10, 479]], dtype=np.int32)
    # create lane trancking object
    laneo = simplelane(cfg_img, roi_points)

    # set loop rate 
    rate = rospy.Rate(rate_hz)

    while  True:    
        rate.sleep()

    rospy.loginfo('Done. exit now!')    

if __name__ == '__main__':
    main()
