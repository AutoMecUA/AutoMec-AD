#!/usr/bin/env python3

"""
    Script that receives the Twist messages from the controller or android app and caps the linear velocity.
    This is important in case of the android app because the app sends a value between -1 and 1, and to have a constant speed we need to cap the value.
    The topic where the Twist messages are subscribed is defined by the parameter 'twist_input_topic'.
    The topic where the Twist messages are published is defined by the parameter 'twist_tmp_topic'.
"""

# Imports
from functools import partial

import rospy
from geometry_msgs.msg import Twist


def twistMsgCallback(message, **kwargs):
    """
    Twist Callback Function that receives the temporary twist messages and caps its linear velocity.
    Args:
        message (Twist): ROS Twist message.
        kwargs (dict): Dictionary with the configuration.
    """
    linear = float(message.linear.x)

    if linear > 0:
        message.linear.x = float(kwargs['linear_velocity'])
    elif linear < 0:
        message.linear.x = -float(kwargs['linear_velocity'])
    else:
        message.linear.x = 0

    # Publish messages
    kwargs['PubTwist'].publish(message)


def main():

    # Initiates the node
    rospy.init_node('twist_cap', anonymous=False)

    # Get parameters
    velocity = rospy.get_param('/linear_velocity', '1') 
    twist_input_topic = rospy.get_param('~twist_input_topic', '/cmd_vel') 
    twist_tmp_topic = rospy.get_param('~twist_tmp_topic', '/cmd_vel_tmp') 

    # Define initial variables
    kwargs: dict[str, object] = dict(PubTwist=None, linear_velocity=velocity)

    # Define publishers
    kwargs["PubTwist"] = rospy.Publisher(twist_tmp_topic, Twist, queue_size=10)

    # Partials
    twistMsgCallback_part = partial(twistMsgCallback, **kwargs)

    # Define subscriber
    rospy.Subscriber(twist_input_topic, Twist, twistMsgCallback_part)

    rospy.spin()


if __name__ == '__main__':
    main()
