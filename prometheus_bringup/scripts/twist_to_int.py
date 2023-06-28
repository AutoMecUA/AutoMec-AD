#!/usr/bin/env python3

# Imports
from functools import partial

import rospy
from std_msgs.msg import Int16
from geometry_msgs.msg import Twist


def twistMsgCallback(message, **kwargs):
    """
    Twist Callback Function
    """
    linear = float(message.linear.x)
    angular = float(message.angular.z)

    if angular < 0:
        angle_out = kwargs['ma1'] * angular + kwargs['ba1']
    else:
        angle_out = kwargs['ma2'] * angular + kwargs['ba2']

    if linear == 1:
        vel = kwargs['vel_max']
    elif linear == -1:
        vel = -kwargs['vel_max']
    else:
        vel = kwargs['vel_center']

    # Publish messages
    kwargs['PubDir'].publish(int(angle_out))
    kwargs['PubVel'].publish(vel)


def main():

    # Initiates the node
    rospy.init_node('twist_to_int', anonymous=False)

    # Get parameters
    twist_cmd_topic = rospy.get_param('~twist_cmd_topic', '/ackermann_steering_controller/cmd_vel') 
    int_dir_topic = rospy.get_param('~int_dir_topic', '/pub_dir')
    int_vel_topic = rospy.get_param('~int_vel_topic', '/pub_vel')
    int_vel_max = rospy.get_param('~int_vel_max', 70)

    # Define initial variables
    kwargs: dict[str, object] = dict(
        vel_center=None, vel_max=None,
        PubDir=None, PubVel=None,
        ma1=None, ba1=None, ma2=None, ba2=None
    )

    # Define publishers
    kwargs['PubDir'] = rospy.Publisher(int_dir_topic, Int16, queue_size=10)
    kwargs['PubVel'] = rospy.Publisher(int_vel_topic, Int16, queue_size=10)

    # Angle
    ang_max = 90+30
    ang_center = 90
    ang_min = 90-30

    kwargs['ma1'] = (ang_center - ang_max) / (0 + 1)
    kwargs['ba1'] = ang_max - int(kwargs['ma1']) * -1

    kwargs['ma2'] = (ang_min - ang_center) / (1 - 0)
    kwargs['ba2'] = ang_center - int(kwargs['ma2']) * 0

    # Velocity
    kwargs['vel_max'] = int_vel_max
    kwargs['vel_center'] = 90
    
    # Partials
    twistMsgCallback_part = partial(twistMsgCallback, **kwargs)

    # Define subscriber
    rospy.Subscriber(twist_cmd_topic, Twist, twistMsgCallback_part)

    rospy.spin()


if __name__ == '__main__':
    main()
