#!/usr/bin/env python3

# Imports
from functools import partial

import rospy
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy


# Direction Callback Function
def messageReceivedCallbackJoy(message, **kwargs):

    velocity_bool = bool(message.buttons[1])
    angular_float = message.axes[0]
    angular_twist = Twist()
    angular_twist.angular.z = angular_float
    kwargs["direction_publisher"].publish(angular_twist)
    kwargs["velocity_publisher"].publish(velocity_bool)


def main():
    kwargs = dict(direction_publisher=None, velocity_publisher=None)

    # Initiating node
    rospy.init_node('joystick_to_android', anonymous=False)

    # Get parameters
    twist_dir_topic = rospy.get_param('~twist_dir_topic', '/android_input_dir') 
    bool_btn_topic = rospy.get_param('~bool_btn_topic', '/android_input_velin')
    joy_topic = rospy.get_param('~joy_topic', '/joy')

    # Define partials
    messageReceivedCallbackJoy_part = partial(messageReceivedCallbackJoy, **kwargs)

    # Define publishers and subscribers
    kwargs["direction_publisher"] = rospy.Publisher(twist_dir_topic, Twist, queue_size=10)
    kwargs["velocity_publisher"] = rospy.Publisher(bool_btn_topic, Bool, queue_size=10)
    rospy.Subscriber(joy_topic, Joy, messageReceivedCallbackJoy_part)
    rospy.spin()


if __name__ == '__main__':
    main()
