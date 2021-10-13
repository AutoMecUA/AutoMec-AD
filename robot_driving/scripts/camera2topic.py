#!/usr/bin/env python3

import cv2
from sensor_msgs.msg._Image import Image
from cv_bridge.core import CvBridge
import rospy
import signal
import sys


def signal_handler(sig, frame):
    # When everything is done, release the capture
    print('The camera was released')
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)


# Initiate the node
rospy.init_node('physical_camera', anonymous=False)

# Get param
camera_topic = rospy.get_param('~camera_topic','/real_camera')
int_camera_id = rospy.get_param('~int_camera_id', 2)

# Define the camera
cap = cv2.VideoCapture(int(int_camera_id))

# Define publisher and rate
Pub = rospy.Publisher(camera_topic, Image, queue_size=10)
rate = rospy.Rate(30)

# set handler on termination
signal.signal(signal.SIGINT, signal_handler)

while not rospy.is_shutdown():
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    #cv2.imshow('frame',frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    # Change the cv2 image to imgmsg and publish
    msg_frame = CvBridge().cv2_to_imgmsg(frame, "bgr8") 
    Pub.publish(msg_frame)

    rate.sleep()


