#!/usr/bin/env python3
# ISTO É PARA O CARRO REAL O STICHER 


import cv2
from sensor_msgs.msg._Image import Image
from cv_bridge.core import CvBridge
import rospy
from stitcherFunction import stitching

# Initiate the node
rospy.init_node('physical_camera', anonymous=False)

# Get param
camera_topic = rospy.get_param('~camera_topic','/real_camera')

int_camera_idL = rospy.get_param('~int_camera_id', 0)
int_camera_idR = rospy.get_param('~int_camera_id', 1)
## TODO : Get 2 int camera_id 

# Define the camera
capL = cv2.VideoCapture(int(int_camera_idL))
capR = cv2.VideoCapture(int(int_camera_idR))

# Define publisher and rate
Pub = rospy.Publisher(camera_topic, Image, queue_size=10)
rate = rospy.Rate(30)

while not rospy.is_shutdown():
    # Capture frame-by-frame
    retL, frameL = capL.read()
    retR, frameR = capR.read()


    frame = stitching(frameL,frameR)

    # Display the resulting frame
    cv2.imshow('frame',frameL)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Change the cv2 image to imgmsg and publish

    msg_frame = CvBridge().cv2_to_imgmsg(frame, "bgr8") 
    Pub.publish(msg_frame)

    rate.sleep()

# When everything is done, release the capture
capL.release()
capR.release()

cv2.destroyAllWindows()
