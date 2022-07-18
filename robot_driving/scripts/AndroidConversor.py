#!/usr/bin/env python3

# Imports
from functools import partial

import rospy
from std_msgs.msg import Int16, Bool
from geometry_msgs.msg import Twist


def messageReceivedCallbackDir(message, ma1, ba1, ma2, ba2, PubDir):
    """Direction Callback Function

    """

    angular = float(message.angular.z)

    if angular < 0:
        angle_out = ma1 * angular + ba1
    else:
        angle_out = ma2 * angular + ba2

        # Publish messages
    PubDir.publish(int(angle_out))


def messageReceivedCallbackBtn(message, **kwargs):
    """Velocity Callback Function

    """

    bool_button = message.data

    # If the button is pressed, bool_vel switches from True to False and viceversa
    if bool_button:
        kwargs["bool_vel"] = not kwargs["bool_vel"]

    kwargs["PubBool"].publish(kwargs["bool_vel"])


def messageReceivedCallbackVel(message, **kwargs):
    """Velocity Callback Function

    """

    bool_cmd = message.data

    # If android_input_vel is true, velocity is max. If not, velocity is zero
    if bool_cmd:
        vel = kwargs["vel_max"]
    else:
        vel = kwargs["vel_center"]

    kwargs["PubVel"].publish(vel)


def main():
    """Program's Core

    """

    # Initiates the node
    rospy.init_node('AndroidConversor', anonymous=False)

    # Get parameters
    twist_dir_topic = rospy.get_param('~twist_dir_topic', '/android_input_dir') 
    vel_cmd_topic = rospy.get_param('~vel_cmd_topic', '/android_input_vel')
    bool_btn_topic = rospy.get_param('~bool_btn_topic', '/android_input_velin')
    int_dir_topic = rospy.get_param('~int_dir_topic', '/pub_dir')
    int_vel_topic = rospy.get_param('~int_vel_topic', '/pub_vel')
    int_vel_max = rospy.get_param('~int_vel_max', 108)

    # Define initial variables
    kwargs: dict[str, object] = dict(
        bool_vel=False,
        vel_center=None, vel_max=None,
        PubDir=None, PubVel=None, PubBool=None,
        ma1=None, ba1=None, ma2=None, ba2=None
    )

    # Global variables
    # Partials
    messageReceivedCallbackBtn_part = partial(messageReceivedCallbackBtn, **kwargs)
    messageReceivedCallbackVel_part = partial(messageReceivedCallbackVel, **kwargs)
    messageReceivedCallbackDir_part = partial(messageReceivedCallbackDir, **kwargs)

    # Define publishers and subscribers
    kwargs["PubDir"] = rospy.Publisher(int_dir_topic, Int16, queue_size=10)
    kwargs["PubVel"] = rospy.Publisher(int_vel_topic, Int16, queue_size=10)
    kwargs["PubBool"] = rospy.Publisher(vel_cmd_topic, Bool, queue_size=10)
    rospy.Subscriber(twist_dir_topic, Twist, messageReceivedCallbackDir_part)
    rospy.Subscriber(vel_cmd_topic, Bool, messageReceivedCallbackVel_part)
    rospy.Subscriber(bool_btn_topic, Bool, messageReceivedCallbackBtn_part)

    # Angle
    # 2 lines
    ang_max = 90+30
    ang_center = 90
    ang_min = 90-30

    kwargs["ma1"] = (ang_center - ang_max) / (0 + 1)
    kwargs["ba1"] = ang_max - int(kwargs["ma1"]) * -1

    kwargs["ma2"] = (ang_min - ang_center) / (1 - 0)
    kwargs["ba2"] = ang_center - int(kwargs["ma2"]) * 0

    # Velocity
    kwargs["vel_max"] = int_vel_max
    kwargs["vel_center"] = 90
    
    rospy.spin()


if __name__ == '__main__':
    main()
