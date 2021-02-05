#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String,Int16
from random import  randrange

def talker():
    pub = rospy.Publisher('pub_dir', Int16, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(2) # 10hz
    while not rospy.is_shutdown():
        # hello_str = "hello world %s" % rospy.get_time()
        rospy.loginfo(randrange(170))
        # rospy.loginfo(hello_str)
        pub.publish(randrange(170))
        # pub.publish(hello_str)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass