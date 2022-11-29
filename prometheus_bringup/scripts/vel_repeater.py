#!/usr/bin/env python3


from functools import partial
from typing import Any
from geometry_msgs.msg._Twist import Twist
import rospy


def twistCallBack(message, config):
    config['twist'] = message


def main():
    config: dict[str, Any] = dict(cmd_vel=None)

    # Defining starting values
    config["twist"] = Twist()

    # Init Node
    rospy.init_node('vel_repeater', anonymous=False)
    
    # Getting parameters
    twist_cmd_topic = rospy.get_param('~twist_cmd_topic', '/cmd_vel')
    twist_tmp_topic = rospy.get_param('~twist_tmp_topic', '/cmd_vel_tmp')
    rate_int = rospy.get_param('~rate_int', '30')

    # Partials
    twistCallBack_part = partial(twistCallBack, config=config)

    # Subscribe and publish topics
    rospy.Subscriber(twist_tmp_topic, Twist, twistCallBack_part)
    twist_pub = rospy.Publisher(twist_cmd_topic, Twist, queue_size=10)

    # Frames per second
    rate = rospy.Rate(rate_int)
    
    while not rospy.is_shutdown():

        # To avoid any errors
        twist_pub.publish(config["twist"])

        rate.sleep()


if __name__ == '__main__':
    main()