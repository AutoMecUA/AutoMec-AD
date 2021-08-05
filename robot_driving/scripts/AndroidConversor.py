#!/usr/bin/env python3

# Imports
import rospy
from std_msgs.msg import Int16, Bool
from geometry_msgs.msg import Twist

# Global Variables
global PubDir
global PubVel
global ma1, ma2, ba1, ba2


# Direction Callback Function
def messageReceivedCallbackDir(message):

    angular = float(message.angular.z)

    if angular < 0:

        AngleOut = ma1 * angular + ba1 
    
    else:

        AngleOut = ma2 * angular + ba2 


    #Publicar mensagens
    PubDir.publish(int(AngleOut))
    
        
# Velocity Callback Function
def messageReceivedCallbackVel(message):

    bool = message.data

    # If the button is pressed, the velocity is max, if it's not, it's null
    if bool:
        vel = vel_max
    else:
        vel = vel_center

    PubVel.publish(vel)


        



# Program's Core
def main():

    global PubDir
    global PubVel
    global ma1, ma2, ba1, ba2
    global vel_max, vel_center

    # Initiates the node
    rospy.init_node('AndroidConversor', anonymous=False)

    # Get parameters
    twist_dir_topic = rospy.get_param('~twist_dir_topic', '/android_input_dir') 
    bool_vel_topic = rospy.get_param('~bool_vel_topic', '/android_input_vel') 
    int_dir_topic = rospy.get_param('~int_dir_topic', '/pub_dir') 
    int_vel_topic = rospy.get_param('~int_vel_topic', '/pub_vel')
    int_vel_max = rospy.get_param('~int_vel_max', 108)



    # Define publishers and subscribers

    PubDir = rospy.Publisher(int_dir_topic, Int16, queue_size=10)
    PubVel = rospy.Publisher(int_vel_topic, Int16, queue_size=10)
    rospy.Subscriber(twist_dir_topic, Twist, messageReceivedCallbackDir)
    rospy.Subscriber(bool_vel_topic, Bool, messageReceivedCallbackVel)

    #Angle

    #2 lines

    ang_max = 90+90/4
    ang_center = 90
    ang_min = 90-90/4

    ma1 = (ang_center - ang_max) / (0 + 1)
    ba1 = ang_max - ma1 * -1

    ma2 = (ang_min - ang_center) / (1 - 0)
    ba2 = ang_center - ma2 * 0

    #Velocity

    vel_max = int_vel_max
    vel_center = 90
    
    rospy.spin()


if __name__ == '__main__':
    main()





