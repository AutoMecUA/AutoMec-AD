#!/usr/bin/env python3

# Imports
from functools import partial

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy


# Direction Callback Function
def messageReceivedCallbackJoy(message, **kwargs):

    angular = message.axes[0]
    linear = round(message.axes[1], 1)
    twist_cmd = Twist()
    twist_cmd.linear.x = linear
    twist_cmd.angular.z = angular
    kwargs["twist_publisher"].publish(twist_cmd)


def main():
    # Defining variables
    kwargs = dict(twist_publisher=None)

    # Initiating node
    rospy.init_node('joystick_to_android', anonymous=False)

    # Get parameters
    twist_cmd_topic = rospy.get_param('~twist_cmd_topic', '/cmd_vel') 
    joy_topic = rospy.get_param('~joy_topic', '/joy')


    # Define publishers and subscribers
    kwargs["twist_publisher"] = rospy.Publisher(twist_cmd_topic, Twist, queue_size=10)
    
    # Define partials
    messageReceivedCallbackJoy_part = partial(messageReceivedCallbackJoy, **kwargs)
    rospy.Subscriber(joy_topic, Joy, messageReceivedCallbackJoy_part)
    rospy.spin()


if __name__ == '__main__':
    main()
