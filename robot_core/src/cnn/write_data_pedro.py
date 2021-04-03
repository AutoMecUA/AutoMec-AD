#!/usr/bin/env python

# Imports
import os.path
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
import time
import pathlib
import os
import string

# Global Variables
global angular
global linear
global bridge
global begin_cmd
global begin_img
global img_rbg
global d
global row


# simple version for working with CWD

s = str(pathlib.Path(__file__).parent.absolute())
path, dirs, files = next(os.walk(s+"/data/IMG/"))
file_count = len(files)
print('Number of images in Already in Folder: ', file_count)


# Function to append row on a csv file
d = file_count
row = []


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


def save_IMG():

    # TODO: Daniel, Este código se for corrido duas vezes vai re-escrever as imagens já guardadas, não estou a conseguir pensar numa solução para guardar o index d
    global d  # index
    global img_rbg
    global angular
    global row
    time.sleep(10/60)  # x/Heartz

    # print('SAVING IMAGE ', d)

    # Define Image Path
    s = str(pathlib.Path(__file__).parent.absolute())
    filename = "file_%d.jpg" % d
    file_path = s+'/data/IMG/' + filename

    # Save Image
    cv2.imwrite(file_path, img_rbg)
    # Add path and steering angle
    row = [file_path, angular]

    d += 1


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
    global d

    # Initial Value
    begin_cmd = False
    begin_img = False
    first_time = True

    # Init Node
    rospy.init_node('write_csv2', anonymous=False)

    # Subscribe topics
    rospy.Subscriber('robot/cmd_vel', Twist, messageReceivedCallback)
    rospy.Subscriber('/robot/camera/rgb/image_raw',
                     Image, message_RGB_ReceivedCallback)

    # Create an object of the CvBridge class
    bridge = CvBridge()
    rate = rospy.Rate(10)  # Alterar este valor não faz nada , falar com daniel

    while True:

        if begin_cmd == False or begin_img == False:
            continue

        # Save Images
        save_IMG()

        if first_time:
            # Create csv file
            # header = ['Center', 'steering']
            now = datetime.now()  # current date and time
            time_now = now.strftime("%H_%M_%S")
            csv_name = now.strftime(
                "%d") + "_" + now.strftime("%m") + "_" + now.strftime("%y") + "__" + time_now
            csv_name += '20_20'
            csv_name += ".csv"
            csv_name = "driving_log.csv"  # Overwrite Name - Falar com daniel sobre isto
            # header comentado para não ter que o filtrar mais tarde
            append_list_as_row(csv_name, row)
            print("File Created")
        else:
            append_list_as_row(csv_name, row)
            print("Row Added", 'n', d)

        first_time = False

        rate.sleep()


if __name__ == '__main__':
    main()
