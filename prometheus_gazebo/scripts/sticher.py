#!/usr/bin/env python3

# Imports
import cv2
import rospy
from sensor_msgs.msg._Image import Image
from cv_bridge.core import CvBridge

# Global Variables
global bridge
global img_rbg_L
global img_rbg_R

global newImageR
global newImageL

# Recieves ImageLeft, Image Right as input and dimesions to resize it , default = (640,480)
# Example:
# left = cv2.imread('Left.png',cv2.IMREAD_COLOR)
# right = cv2.imread('Right.png',cv2.IMREAD_COLOR)
# dim = (640, 480)

def stitching(left,right,dim=(640,480)):
    # Nota. as vezes funciona melhor com o resize
    left = cv2.resize(left,dim,interpolation = cv2.INTER_AREA)
    right = cv2.resize(right,dim,interpolation = cv2.INTER_AREA)

    images = []

    images.append(left)
    images.append(right)

    # cv2.Stitcher_SCANS // cv2.Stitcher_PANORAMA
    stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
    ret, pano = stitcher.stitch(images)

    return ret, pano


# Callback function to receive image
def message_RGB_ReceivedCallbackL(message):
    global img_rbg_L
    global bridge
    global newImageL
    if not newImageL:
        img_rbg_L = bridge.imgmsg_to_cv2(message, "bgr8")
        newImageL = True

def message_RGB_ReceivedCallbackR(message):
    global img_rbg_R
    global bridge
    global newImageR
    if not newImageR:
        img_rbg_R = bridge.imgmsg_to_cv2(message, "bgr8")
        newImageR = True

def main():
    # Global variables
    global bridge
    global img_rbg_L
    global img_rbg_R
    global newImageR
    global newImageL

    newImageL = False
    newImageR = False

    # Init Node
    rospy.init_node('stitching_imgs', anonymous=False)

    camera_topic = rospy.get_param('~camera_topic','/ackermann_vehicle/camerapan/rgb/image_raw')
    image_raw_topic = rospy.get_param('~image_raw_topic', '/ackermann_vehicle/cameral/rgb/image_raw') 
    image_raw_topic_R = rospy.get_param('~image_raw_topic', '/ackermann_vehicle/camerar/rgb/image_raw') 

    rate_hz = rospy.get_param('~rate', 10)
    image_width = rospy.get_param('~width', 640)
    image_height = rospy.get_param('~height', 480)

    # Subscribe topics
    rospy.Subscriber(image_raw_topic, Image, message_RGB_ReceivedCallbackL)
    rospy.Subscriber(image_raw_topic_R, Image, message_RGB_ReceivedCallbackR)
    Pub = rospy.Publisher(camera_topic, Image, queue_size=10)

    # Create an object of the CvBridge class
    bridge = CvBridge()

    # set loop rate 
    rate = rospy.Rate(rate_hz)

    while  True:
        
        if newImageL and newImageR:

            ret, frame = stitching(img_rbg_L,img_rbg_R)
            
            if ret == cv2.STITCHER_OK:
                # resize
                frame = cv2.resize(frame, (image_width, image_height), interpolation = cv2.INTER_AREA)
                # convert  publish
                msg_frame = CvBridge().cv2_to_imgmsg(frame, "bgr8") 
                Pub.publish(msg_frame)

            newImageL = newImageR = False

        rate.sleep()

if __name__ == '__main__':
    main()