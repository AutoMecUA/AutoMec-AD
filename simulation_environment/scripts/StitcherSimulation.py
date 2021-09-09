#!/usr/bin/env python3

# Imports

import cv2
import rospy

from sensor_msgs.msg._Image import Image
from cv_bridge.core import CvBridge








# CARRO VIRTUAL


# Recieves ImageLeft, Image Right as input and dimesions to resize it , default = (640,480)
# Example:
#  left = cv2.imread('Left.png',cv2.IMREAD_COLOR)
# right = cv2.imread('Right.png',cv2.IMREAD_COLOR)
# dim = (640, 480)

def stitching(left,right,dim=(640,480)):

    # Nota. as vezes funciona melhor com o resize
    left = cv2.resize(left,dim,interpolation = cv2.INTER_AREA)
    right = cv2.resize(right,dim,interpolation = cv2.INTER_AREA)


    images = []

    images.append(left)
    images.append(right)

    stitcher = cv2.Stitcher.create()
    ret,pano = stitcher.stitch(images)

    return pano
    #
    #if ret == cv2.STITCHER_OK:
    #    cv2.imshow('Panorama',pano)
    #    cv2.waitKey()
    #   cv2.destroyAllWindows()
    #else:
    #    print('Error during stitching')



# Global Variables
global bridge
global img_rbg
global img_rbg_R

global newImageR
global newImageL


# Callback function to receive image
def message_RGB_ReceivedCallback(message):
    
    global img_rbg
    global bridge
    global newImageL

    img_rbg = bridge.imgmsg_to_cv2(message, "bgr8")

    newImageL = True

def message_RGB_ReceivedCallbackR(message):
    
    global img_rbg_R
    global bridge
    global newImageR
    

    img_rbg_R = bridge.imgmsg_to_cv2(message, "bgr8")

    newImageR= True



def main():

    # Global variables
  
    global bridge
    global img_rbg
    global img_rbg_R
    global newImageR
    global newImageL


    # Initial Value

    # Init Node
    rospy.init_node('stitching_data', anonymous=False)

    camera_topic = rospy.get_param('~camera_topic','/real_cameraPANO')
    image_raw_topic = rospy.get_param('~image_raw_topic', '/ackermann_vehicle/cameral/rgb/image_raw') 
    image_raw_topic_R = rospy.get_param('~image_raw_topic', '/ackermann_vehicle/camerar/rgb/image_raw') 

    rate_hz = rospy.get_param('~rate', 30)
    image_width = rospy.get_param('~width', 320)
    image_height = rospy.get_param('~height', 160)


    
    # Subscribe topics
    rospy.Subscriber(image_raw_topic, Image, message_RGB_ReceivedCallback)
    rospy.Subscriber(image_raw_topic_R, Image, message_RGB_ReceivedCallbackR)
    Pub = rospy.Publisher(camera_topic, Image, queue_size=10)


    # Create an object of the CvBridge class
    bridge = CvBridge()

       
    # set loop rate 
    rate = rospy.Rate(rate_hz)


    while  True:
        
        #if not (newImageL and newImageR):
         #   continue

        frame = stitching(img_rbg,img_rbg_R)
        # save image L
        #dim = (image_width, image_height)
        #img_rbg = cv2.resize(img_rbg, dim, interpolation = cv2.INTER_AREA)
        #image_saved = Image_pil.fromarray(img_rbg)
        
        msg_frame = CvBridge().cv2_to_imgmsg(frame, "bgr8") 
        Pub.publish(msg_frame)

        rate.sleep()

if __name__ == '__main__':
    main()