#!/usr/bin/env python

# Imports
import rospy
from std_msgs.msg import Int16
from geometry_msgs.msg import Twist

# Global Variables
global PubDir
global PubVel
global ma1, ma2, ba1, ba2, mv1, bv1, mv2, bv2
global Backward, Forward, BreakMode


# Direction Callback Function
def messageReceivedCallbackDir(message):

    global PubDir
    global PubVel
    global Backward, Forward, BreakMode

    angular = float(message.angular.z)



    #--------------------------------------------------------------------------
    #-----------------------------Angular--------------------------------------
    #--------------------------------------------------------------------------


    if angular < 0:

        AngleOut = ma1 * angular + ba1 
    
    else:

        AngleOut = ma2 * angular + ba2 


   
    #Publicar mensagens
    

    PubDir.publish(int(AngleOut))
    


    if BreakMode:
        rospy.sleep(0.5)
        BreakMode=False
    else:
        pass
        


# Velocity Callback Function
def messageReceivedCallbackVel(message):

    global PubDir
    global PubVel
    global Backward, Forward, BreakMode

    linear = float(message.linear.x)

    #--------------------------------------------------------------------------
    #-----------------------------Velocity-------------------------------------
    #--------------------------------------------------------------------------


    
    #Code to when going backward (coming from forward), pause 2 seconds, and then go

    if linear > 0:
        Backward = False
        BreakMode = False
    else:
        if Backward == False:
            linear = 0
            Backward = True
            BreakMode = True
        else:
            BreakMode = False
        

        
    #Code to when going forward (coming from backward), pause 2 seconds, and then go



    if not BreakMode:
        if linear < 0:
            Forward = False
            BreakMode = False
        else:
            if Forward == False:
                linear = 0
                Forward = True
                BreakMode = True
            else:
                BreakMode = False
            


    if linear <0:
        VelOut = mv1 * linear + bv1
    else:
        
        VelOut = mv2 * linear + bv2




    
    #Publicar mensagens
    

    PubVel.publish(int(VelOut))
    


    if BreakMode:
        rospy.sleep(0.5)
        BreakMode=False
    else:
        pass
        



# Program's Core
def main():

    global PubDir
    global PubVel
    global ma1, ma2, ba1, ba2, mv1, bv1, mv2, bv2
    global BreakMode, Forward, Backward
    global PubDir, PubVel

    BreakMode = False
    Forward = False
    Backward = False

    


    rospy.init_node('AndroidConversor', anonymous=False)

    twist_dir_topic = rospy.get_param('~twist_dir_topic', '/android_input_dir') 
    twist_vel_topic = rospy.get_param('~twist_vel_topic', '/android_input_vel') 
    int_dir_topic = rospy.get_param('~int_dir_topic', '/pub_dir') 
    int_vel_topic = rospy.get_param('~int_vel_topic', '/pub_vel')
    int_vel_max = rospy.get_param('~int_vel_max', 125)




    PubDir = rospy.Publisher(int_dir_topic, Int16, queue_size=10)
    PubVel = rospy.Publisher(int_vel_topic, Int16, queue_size=10)
    rospy.Subscriber(twist_dir_topic, Twist, messageReceivedCallbackDir)
    rospy.Subscriber(twist_vel_topic, Twist, messageReceivedCallbackVel)

    #Angle

    #2 lines

    ang_max = 170
    ang_center = 90
    ang_min = 0

    ma1 = (ang_center - ang_max) / (0 + 1)
    ba1 = ang_max - ma1 * -1

    ma2 = (ang_min - ang_center) / (1 - 0)
    ba2 = ang_center - ma2 * 0

    #Velocity

    #2 line



    vel_max = int_vel_max
    vel_center = 90
    vel_min = 0

    mv1 = (vel_min - vel_center) / (-1)
    bv1 = vel_center 


    mv2 = (vel_max - vel_center)
    bv2 = vel_center
    
    rospy.spin()


if __name__ == '__main__':
    main()





