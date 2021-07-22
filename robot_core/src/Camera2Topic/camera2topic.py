#!/usr/bin/env python3

import numpy as np
import cv2
from sensor_msgs.msg._Image import Image
from cv_bridge.core import CvBridge
import rospy

cap = cv2.VideoCapture(2)
rospy.init_node('camera2ros_topic', anonymous=False)
Pub = rospy.Publisher('real_camera', Image, queue_size=10)
rate = rospy.Rate(10)
while not rospy.is_shutdown():
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    # cv2.imshow('frame',frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    msg_frame = CvBridge().cv2_to_imgmsg(frame, "bgr8") 
    Pub.publish(msg_frame)

    rate.sleep()

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
