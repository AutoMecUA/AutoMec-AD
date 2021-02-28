#!/usr/bin/env python

# Imports
import argparse
import cv2
from csv import writer
import copy
import numpy as np
import rospy
from geometry_msgs.msg._Twist import Twist
from sensor_msgs.msg._Image import Image
from cv_bridge.core import CvBridge
from datetime import datetime

# Global Variables
global angular
global linear
global bridge
global begin_cmd
global begin_img
global img_rbg

# Function to append row on a csv file
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)
        write_obj.close()

# Calback Function to receive the cmd values
def messageReceivedCallback(message):

    global angular
    global linear
    global begin_cmd

    angular = float(message.angular.z)
    linear = float(message.linear.x)

    begin_cmd = True

# Callback function to receive image
def message_RGB_ReceivedCallback(message):
    
    global img_rbg
    global bridge
    global begin_img


    img_rbg = bridge.imgmsg_to_cv2(message, "bgr8")

    begin_img = True

    

def main():

    # Global variables
    global angular
    global linear
    global bridge
    global img_rbg
    global begin_cmd
    global begin_img

    # Initial Value
    begin_cmd = False
    begin_img = False
    scale_percent = 25 # percent of original size
    first_time = True

    # Init Node
    rospy.init_node('write_csv', anonymous=False)

    # Subscribe topics
    rospy.Subscriber('robot/cmd_vel', Twist, messageReceivedCallback)
    rospy.Subscriber('/robot/camera/rgb/image_raw', Image, message_RGB_ReceivedCallback)
    
    # Create an object of the CvBridge class
    bridge = CvBridge()

    rate = rospy.Rate(30)
    while  True:
        
        if begin_cmd==False or begin_img==False:
            continue

        # Gray scale the image
        img_gray = cv2.cvtColor(img_rbg,cv2.COLOR_BGR2GRAY)

        # Binarize the image
        _, img_tresh = cv2.threshold(img_gray, 127, 1, cv2.THRESH_BINARY)

        # Resizing image
        width = int(img_tresh.shape[1] * scale_percent / 100)
        height = int(img_tresh.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img_tresh, dim, interpolation = cv2.INTER_AREA)
    

        # Tranform image in a list
        initial_array = copy.deepcopy(resized)
        final_array=np.resize(initial_array,(1,initial_array.shape[0]*initial_array.shape[1]))
        final_list = final_array.tolist()[0]

      
        # Append the speed and the angle of the cmd_topic
        final_list.append(linear)
        final_list.append(angular)

     
        if first_time:
            # Create csv file
            header = ["pixel"] * len(final_list)
            header.append("linear")
            header.append("angular")
            now = datetime.now() # current date and time
            time_now = now.strftime("%H_%M_%S")
            csv_name = now.strftime("%d") + "_" + now.strftime("%m") + "_" + now.strftime("%y") + "__" + time_now
            csv_name += ".csv"
            append_list_as_row(csv_name, header)
            print("File Created")

        else:
            append_list_as_row(csv_name, final_list)
            print("Row Added")
            
        
        first_time=False
        
        rate.sleep()


    
    


if __name__ == '__main__':
    main()