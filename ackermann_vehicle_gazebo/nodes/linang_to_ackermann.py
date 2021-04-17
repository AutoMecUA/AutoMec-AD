#!/usr/bin/env python3
# Author: sergio.inacio@ua.pt (inaciose@gmail.com)

import rospy, math
from geometry_msgs.msg import Twist
from ackermann_msgs.msg import AckermannDriveStamped
from ackermann_msgs.msg import AckermannDrive

def convert_trans_rot_vel_to_steering_angle(v, omega, wheelbase):
  if omega == 0 or v == 0:
    return 0

  radius = v / omega
  return math.atan(wheelbase / radius)

def vel_cmd_callback(data):
  global vlinear
  vlinear = data.linear.x

def angle_cmd_callback(data):
  global vangular
  vangular = data.angular.z

if __name__ == '__main__': 
  try:
    
    rospy.init_node('linang_to_ackermann_drive')
    
    global vlinear
    global vangular

    # default velocities
    vlinear = 0
    vangular = 0

    # Set variables from parameters 
    vel_topic = rospy.get_param('~vel_topic', '/vel_cmd')
    angle_topic = rospy.get_param('~angle_topic', '/angle_cmd') 
    ackermann_cmd_topic = rospy.get_param('~ackermann_cmd_topic', '/ackermann_cmd')
    wheelbase = rospy.get_param('~wheelbase', 1.0)
    frame_id = rospy.get_param('~frame_id', 'odom')
    message_type = rospy.get_param('~message_type', 'ackermann_drive') # ackermann_drive or ackermann_drive_stamped
    
    # Set subscribers and publishers
    rospy.Subscriber(vel_topic, Twist, vel_cmd_callback, queue_size=1)
    rospy.Subscriber(angle_topic, Twist, angle_cmd_callback, queue_size=1)
    if message_type == 'ackermann_drive':
      pub = rospy.Publisher(ackermann_cmd_topic, AckermannDrive, queue_size=1)
    else:
      pub = rospy.Publisher(ackermann_cmd_topic, AckermannDriveStamped, queue_size=1)
    
    ###################

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():

      steering = convert_trans_rot_vel_to_steering_angle(vlinear, vangular, wheelbase)

      if message_type == 'ackermann_drive':

        msg = AckermannDrive()
        msg.steering_angle = steering
        msg.speed = vlinear
        
      else:

        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = frame_id
        msg.drive.steering_angle = steering
        msg.drive.speed = vlinear
        
      pub.publish(msg)
    
  except rospy.ROSInterruptException:
    pass

