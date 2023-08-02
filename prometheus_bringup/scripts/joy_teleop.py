#!/usr/bin/env python3

"""
    Script to establish a connection between the xbox controller and the car that publishes the commands to the topic '/cmd_vel'.
    The topic to publish the commands can be changed by setting the parameter 'twist_cmd_topic'.
    The topic to subscribe to the xbox controller can be changed by setting the parameter 'joy_topic'.
"""

# Imports
from functools import partial

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy


# Direction Callback Function
def messageReceivedCallbackJoy(message, **kwargs):
    """
    Callback function for the controller that receives the joy message and publishes to the car.
    Args:
        message (Joy): ROS Joy message.
        kwargs (dict): Dictionary with the configuration.
    """

    angular = message.axes[0]/3

    # The R2 trigger rest's at 1, and goes up to -1 as its pressed
    if(message.axes[4] < 0): 
        linear = round(abs(message.axes[4]), 1) # For scaling vel, linear belongs in [0,1]
    elif message.axes[3] > 0:
        linear = 1
    elif message.axes[3] < 0:
        linear = -1
    else:
        linear = 0

    twist_cmd = Twist()  # Message type twist
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
