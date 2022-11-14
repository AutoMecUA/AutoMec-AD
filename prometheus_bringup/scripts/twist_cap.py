#!/usr/bin/env python3

# Imports
from functools import partial

import rospy
from geometry_msgs.msg import Twist


def twistMsgCallback(message, **kwargs):
    """
    Twist Callback Function
    """
    linear = float(message.linear.x)

    if linear > 0:
        message.linear.x = float(kwargs['linear_velocity'])

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
    twist_cmd_topic = rospy.get_param('~twist_cmd_topic', '/ackermann_steering_controller/cmd_vel') 

    # Define initial variables
    kwargs: dict[str, object] = dict(PubTwist=None, linear_velocity=velocity)

    # Define publishers
    kwargs["PubTwist"] = rospy.Publisher(twist_cmd_topic, Twist, queue_size=10)

    # Partials
    twistMsgCallback_part = partial(twistMsgCallback, **kwargs)

    # Define subscriber
    rospy.Subscriber(twist_input_topic, Twist, twistMsgCallback_part)

    rospy.spin()


if __name__ == '__main__':
    main()
