#!/usr/bin/env python

# Imports
import rospy
from std_msgs.msg import Int16
from geometry_msgs.msg import Twist

# Global Variables
global PubDir
global PubVel
global ma1, ma2, ba1, ba2, mv1, bv1
global Backward, Forward, BreakMode


# Calback Function
def messageReceivedCallback(message):

    global PubDir
    global PubVel
    global Backward, Forward, BreakMode

    angular = float(message.angular.z)
    linear = float(message.linear.x)



    #--------------------------------------------------------------------------
    #-----------------------------Angular--------------------------------------
    #--------------------------------------------------------------------------


    if angular < 0:

        AngleOut = ma1 * angular + ba1 
    
    else:

        AngleOut = ma2 * angular + ba2 


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
            


    VelOut = mv1 * linear + bv1



    
    #Publicar mensagens
    

    PubDir.publish(int(AngleOut))
    PubVel.publish(int(VelOut))
    


    if BreakMode:
        rospy.sleep(1)
        BreakMode=False
    else:
        pass
        

    #print("Velocidade Linear " + str(message.linear.x))
    #print("Angulo Z " + str(message.angular.z))

    
# Program's Core
def main():

    global PubDir
    global PubVel
    global ma1, ma2, ba1, ba2, mv1, bv1
    global BreakMode, Forward, Backward
    global PubDir, PubVel

    BreakMode = False
    Forward = False
    Backward = False


    rospy.init_node('AndroidConversor', anonymous=False)
    PubDir = rospy.Publisher('pub_dir', Int16, queue_size=10)
    PubVel = rospy.Publisher('pub_vel', Int16, queue_size=10)
    rospy.Subscriber('android_input', Twist, messageReceivedCallback)

    #Angle

    #2 lines
    ma1 = (90 - 170) / (0 + 1)
    ba1 = 170 - ma1 * -1

    ma2 = (0 - 90) / (1 - 0)
    ba2 = 90 - ma2 * 0

    #Velocity

    #1 line

    mv1 = (90 - 0) / (0 + 1)
    bv1 = 0 - mv1 * -1
    
    rospy.spin()


if __name__ == '__main__':
    main()





