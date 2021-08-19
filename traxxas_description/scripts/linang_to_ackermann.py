#!/usr/bin/env python3
# Author: sergio.inacio@ua.pt (inaciose@gmail.com)

import rospy, math
from geometry_msgs.msg import Twist
from ackermann_msgs.msg import AckermannDriveStamped
from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import Int16

###################
# CALLBACK & HELPER
###################

def convert_trans_rot_vel_to_steering_angle(v, omega, wheelbase):
  if omega == 0 or v == 0:
    return 0

  radius = v / omega
  return math.atan(wheelbase / radius)

# callbacks used in autonomous mode
def vel_cmd_callback(data):
  global vlinear
  vlinear = data.linear.x
  
def angle_cmd_callback(data):
  global vangular
  vangular = data.angular.z

# callback used in training mode
def cmd_vel_callback(data):
  global vlinear
  global vangular
  vlinear = data.linear.x
  vangular = data.angular.z
  #print(data)

if __name__ == '__main__': 
  try:
    
    rospy.init_node('linang_to_ackermann_drive')
    
    global vlinear
    global vangular

    # default velocities
    vlinear = 0.0
    vangular = 0.0
    
    ###################
    # PARAMS CONFIG
    ###################

    # general config
    message_type = rospy.get_param('~message_type', 'ackermann_drive') # ackermann_drive or ackermann_drive_stamped or phisical
    train = rospy.get_param('~train', 0)
    wheelbase = rospy.get_param('~wheelbase', 1.0)
    frame_id = rospy.get_param('~frame_id', 'odom')
    rate = rospy.get_param('~rate', 20)

    # topics config
    cmd_vel_topic = rospy.get_param('~cmd_vel', '/cmd_vel')       # sub train
    vel_topic = rospy.get_param('~vel_topic', '/vel_cmd')         # sub auto
    angle_topic = rospy.get_param('~angle_topic', '/angle_cmd')   # sub auto 
    pub_dir_topic = rospy.get_param('~pub_dir', '/pub_dir')       # pub phisical 
    pub_vel_topic = rospy.get_param('~pub_vel', '/pub_vel')       # pub phisical
    ackermann_cmd_topic = rospy.get_param('~ackermann_cmd_topic', '/ackermann_cmd') # pub gazebo
    
    # limits
    servo_limit = rospy.get_param('~servo_limit', 30)       # rover steering limit (degrees)
    esc_limit = rospy.get_param('~esc_limit', 120)          # rover velocity limit (esc format)
    max_vlinear = rospy.get_param('~max_vlinear', 2)        # rover max velocity m/s
    
    ###################
    # SETUP
    ###################
        
    print(train)
    # Set subscribers
    if train == 1:
      # train mode callbacks
      print("Training mode")
      rospy.Subscriber(cmd_vel_topic, Twist, cmd_vel_callback, queue_size=1)
    else:
      # autonomous mode callbacks
      print("Autonomous mode")
      rospy.Subscriber(vel_topic, Twist, vel_cmd_callback, queue_size=1)
      rospy.Subscriber(angle_topic, Twist, angle_cmd_callback, queue_size=1)

    # Set publishers
    if message_type == 'ackermann_drive':
      pub = rospy.Publisher(ackermann_cmd_topic, AckermannDrive, queue_size=1)
    elif message_type == 'ackermann_drive_stamped':
      pub = rospy.Publisher(ackermann_cmd_topic, AckermannDriveStamped, queue_size=1)
    else:
      pubd = rospy.Publisher(pub_dir_topic, Int16, queue_size=1)
      pubv = rospy.Publisher(pub_vel_topic, Int16, queue_size=1)
    
    ###################
    # LOOP
    ###################

    rate = rospy.Rate(rate)

    while not rospy.is_shutdown():

      steering = convert_trans_rot_vel_to_steering_angle(vlinear, vangular, wheelbase)

      if message_type == 'ackermann_drive':

        msg = AckermannDrive()
        msg.steering_angle = steering
        msg.speed = vlinear
        pub.publish(msg)
        
      elif message_type == 'ackermann_drive_stamped':

        msg = AckermannDriveStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = frame_id
        msg.drive.steering_angle = steering
        msg.drive.speed = vlinear

        pub.publish(msg)

      else:
        # steering
        msgd = Int16()
        # 1 rad = 57.2957795131
        steering = steering * 57.2957795131

        if steering > servo_limit:
          steering = servo_limit
        elif steering < -servo_limit:
          steering = -servo_limit
        msgd.data = round(steering)

        # velocity
        msgv = Int16()
        wrk_vlinear = vlinear
        if vlinear == 0:
          # stope
          wrk_vlinear = 90
        elif vlinear > 0:
          # forward
          wrk_vlinear = (90 * vlinear) / max_vlinear
          wrk_vlinear = wrk_vlinear + 90
          if wrk_vlinear > esc_limit:
            wrk_vlinear = esc_limit
        else:
          # backward
          wrk_vlinear = (90 * abs(vlinear)) / max_vlinear
          wrk_vlinear = 90 - wrk_vlinear
          if wrk_vlinear < 90 - (esc_limit - 90):
            wrk_vlinear = 90 - (esc_limit - 90)

        msgv.data = round(wrk_vlinear)

        pubd.publish(msgd)
        pubv.publish(msgv)
      
      rate.sleep()

    
  except rospy.ROSInterruptException:
    pass
