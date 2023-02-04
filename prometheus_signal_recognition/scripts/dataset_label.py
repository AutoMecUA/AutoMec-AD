#!/usr/bin/env python3

# Imports
import rospy
from colorama import Fore, Style
from typing import Any
from functools import partial
import numpy as np
from gazebo_msgs.msg._LinkStates import LinkStates
from sensor_msgs.msg._CameraInfo import CameraInfo
import tf
import math

# Callback function to receive link states
def poses_callback(message, config: dict):
 
    config['link_name'] = message.name
    config['link_pose'] = message.pose

# Callback function to receive camera info
def parameter_camera_callback(message, config: dict):
    
    config['parameter_camera'] = message.K
    
# Calculate transformation matrix between two points
def transformation_matrix(point_array_1,point_array_2):
    """calculate transformation matrix between two points
    Args:
    point_array_1: [x,y,z,qx,qy,qz,qw]
    point_array_2: [x,y,z,qx,qy,qz,qw]
    """
    point_array =  [point_array_1,point_array_2]
    
    for number_point, point in enumerate(point_array,start=1):
        
        quaternion1 = (point[3],point[4],point[5],point[6])
        euler = tf.transformations.euler_from_quaternion(quaternion1, axes='sxyz') # will provide result in x, y,z sequence
        roll=euler[0]
        pitch=euler[1]
        yaw=euler[2]

        C00=math.cos(yaw)*math.cos(pitch)
        C01=math.cos(yaw)*math.sin(pitch)*math.sin(roll) - math.sin(yaw)*math.sin(roll)
        C02=math.cos(yaw)*math.sin(pitch)*math.cos(roll) + math.sin(yaw)*math.sin(roll)
        C03=point[0]
        C10=math.sin(yaw)*math.cos(pitch)
        C11=math.sin(yaw)*math.sin(pitch)*math.sin(roll) + math.cos(yaw)*math.cos(roll)
        C12=math.sin(yaw)*math.sin(pitch)*math.cos(roll) -math.cos(yaw)*math.sin(roll)
        C13=point[1]
        C20=-math.sin(pitch)
        C21=math.cos(pitch)*math.sin(roll)
        C22=math.cos(pitch)*math.cos(roll)
        C23=point[2]
        C30=0
        C31=0
        C32=0
        C33=1

        if number_point == 1:
            obj1_mat=np.array([[C00, C01, C02, C03],[C10, C11, C12, C13],[C20, C21, C22, C23],[C30, C31, C32, C33]])
        else:
            obj2_mat=np.array([[C00, C01, C02, C03],[C10, C11, C12, C13],[C20, C21, C22, C23],[C30, C31, C32, C33]])

    transformation_mat=np.dot(np.linalg.inv(obj2_mat), obj1_mat) #generating the transformation
    return transformation_mat


def main():
    # Global variables
    config: dict[str, Any] = dict(
        link_name=None,
        link_pose=None,
        parameter_camera=None
    )

    # print(f'{Fore.RED}Hello world! {Style.RESET_ALL}')

    # Init Node
    rospy.init_node('label_data', anonymous=False)

    # Retrieving parameters
    rate_hz = rospy.get_param('~rate', 30)
  
    # Subscribers
    PosesCallback_part = partial(poses_callback, config=config)
    rospy.Subscriber('gazebo/link_states', LinkStates, PosesCallback_part)
    ParameterCameraCallback_part = partial(parameter_camera_callback, config=config)
    rospy.Subscriber('/top_right_camera/camera_info', CameraInfo, ParameterCameraCallback_part)
    
    # set loop rate
    rate = rospy.Rate(rate_hz)

    while not rospy.is_shutdown():
        signal_poses = []
        camera_poses = []
        if not config['link_name'] is None and not config['link_pose'] is None:
            for i in range(len(config['link_name'])):
                name = config['link_name'][i].split('::')

                if name[1] == 'pictogram':
                    signal_pose = {}
                    # signal_pose[message.name[i]] = message.pose[i]
                    signal_pose['signal'] = name[0]
                    signal_pose['pose'] = config['link_pose'][i]
                    signal_poses.append(signal_pose)

                if name[1] == 'front_right_steer_link':
                    camera_pose = {}
                    camera_pose['signal'] = name[0]
                    camera_pose['pose'] = config['link_pose'][i]
                    camera_poses.append(camera_pose)
            
            # Array of signals
            signal_pose_array = np.array([[pose['pose'].position.x,pose['pose'].position.y,pose['pose'].position.z,pose['pose'].orientation.x,pose['pose'].orientation.y,pose['pose'].orientation.z,pose['pose'].orientation.w] for pose in signal_poses],dtype = np.float64)

            # Array of cameras
            camera_pose_array = np.array([[pose['pose'].position.x,pose['pose'].position.y,pose['pose'].position.z,pose['pose'].orientation.x,pose['pose'].orientation.y,pose['pose'].orientation.z,pose['pose'].orientation.w] for pose in camera_poses],dtype = np.float64)

            #print(signal_pose_array)
            #print(camera_pose_array)

            transformation_mat=transformation_matrix(signal_pose_array[0],camera_pose_array[0])
            
            #print(transformation_mat)
                

            

        rate.sleep()

   
if __name__ == '__main__':
    main()