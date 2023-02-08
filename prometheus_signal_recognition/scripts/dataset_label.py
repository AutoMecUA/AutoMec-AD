#!/usr/bin/env python3

# Imports
import rospy
from colorama import Fore, Style
from typing import Any
from functools import partial
import numpy as np
from scipy.spatial.transform import Rotation as R
from gazebo_msgs.msg._LinkStates import LinkStates
from sensor_msgs.msg._CameraInfo import CameraInfo
import tf
from math import cos, sin, pi
import cv2
from cv_bridge.core import CvBridge
from sensor_msgs.msg._Image import Image

# Callback function to receive link states
def poses_callback(message, config: dict):
 
    config['link_name'] = message.name
    config['link_pose'] = message.pose


# Callback function to receive camera info
def parameter_camera_callback(message, config: dict):
    
    config['camera_matrix'] = message.K
    config['camera_matrix'] = np.reshape(config['camera_matrix'], (3, 3))
    config['camera_matrix'][0][2] = 320
    config['camera_matrix'][1][2] = 240
  

# Callback function to receive image
def imgRgbCallback(message, config: dict):
    
    config['img_rgb'] = config['bridge'].imgmsg_to_cv2(message, "rgb8")


# Convert quaternion to roll, pitch, yaw
def quaternion2rpy(point_array):
    quaternion1 = (point_array[3],point_array[4],point_array[5],point_array[6])
    euler = tf.transformations.euler_from_quaternion(quaternion1, axes='sxyz') # will provide result in x, y,z sequence
    roll=euler[0]
    pitch=euler[1]
    yaw=euler[2]
    return roll, pitch, yaw


# # Calculate transformation matrix between two points
# def transformation_matrix(point_array_1,point_array_2):
#     """calculate transformation matrix between two points
#     Args:
#     point_array_1: [x,y,z,qx,qy,qz,qw]
#     point_array_2: [x,y,z,qx,qy,qz,qw]
#     """
#     point_array =  [point_array_1,point_array_2]
    
#     for number_point, point in enumerate(point_array,start=1):
        
#         roll, pitch, yaw = quaternion2rpy(point)

#         C00=cos(yaw)*cos(pitch)
#         C01=cos(yaw)*sin(pitch)*sin(roll) - sin(yaw)*sin(roll)
#         C02=cos(yaw)*sin(pitch)*cos(roll) + sin(yaw)*sin(roll)
#         C03=point[0]
#         C10=sin(yaw)*cos(pitch)
#         C11=sin(yaw)*sin(pitch)*sin(roll) + cos(yaw)*cos(roll)
#         C12=sin(yaw)*sin(pitch)*cos(roll) -cos(yaw)*sin(roll)
#         C13=point[1]
#         C20=-sin(pitch)
#         C21=cos(pitch)*sin(roll)
#         C22=cos(pitch)*cos(roll)
#         C23=point[2]
#         C30=0
#         C31=0
#         C32=0
#         C33=1

#         if number_point == 1:
#             obj1_mat=np.array([[C00, C01, C02, C03],[C10, C11, C12, C13],[C20, C21, C22, C23],[C30, C31, C32, C33]])
#         else:
#             obj2_mat=np.array([[C00, C01, C02, C03],[C10, C11, C12, C13],[C20, C21, C22, C23],[C30, C31, C32, C33]])

#     transformation_mat=np.dot(np.linalg.inv(obj2_mat), obj1_mat) #generating the transformation
#     return transformation_mat


def get_matrix(point_array,trans,rot):

    r = R.from_quat(rot)
    rot = r.as_matrix()

    # homogeneous transformation matrix (4,4) 
    l = 4
    matrix_footprint2cam = np.zeros((l,l))
    matrix_footprint2cam[0:3,0:3] = rot
    matrix_footprint2cam[0:3,3] = trans
    matrix_footprint2cam[3,3] = 1

    # rotation matrix
    r = R.from_quat([point_array[3], point_array[4], point_array[5], point_array[6]])
    rot_matrix = r.as_matrix()
    
    # translation matrix
    trans_matrix = np.array([point_array[0], point_array[1], point_array[2]], dtype=np.float32)

    # homogeneous transformation matrix (4,4) 
    matrix_world2footprint = np.zeros((l,l))
    matrix_world2footprint[0:3,0:3] = rot_matrix
    matrix_world2footprint[0:3,3] = trans_matrix
    matrix_world2footprint[3,3] = 1

    # print('matrix_world2footprint: \n', matrix_world2footprint)
    # print('matrix_footprint2cam: \n', matrix_footprint2cam)
    # r = R.from_matrix(matrix_footprint2cam[0:3,0:3])
    # print(r.as_euler('zyx', degrees=True))
    
    matrix = np.dot(matrix_world2footprint,matrix_footprint2cam)
    #print(matrix)
    
    # inverse matrices
    matrix_inv = np.linalg.inv(matrix)


    # r = R.from_matrix(matrix_inv[0:3,0:3])
    # print(r.as_euler('zyx', degrees=True))
    rot_matrix = matrix[0:3,0:3]
    trans_matrix = matrix[0:3,3]
    
    # print(rot_matrix)
    #print(trans_matrix)
    return rot_matrix, trans_matrix


# Calculate bounding box for each signal
def get_bounding_box(point):    
    """get bounding box for each signal
    Args:
    point: [x,y,z,qx,qy,qz,qw]
    Returns:
    bounding_box: array with 8 corners of the bounding box
    """

    _, _, yaw = quaternion2rpy(point)
    
    # Since the sign is square, the height is equal to the length.
    height = 0.305
    width = 0.005

    #Translation matrix corner 
    T_sup_left = np.array([cos(yaw)*height,sin(yaw)*height,height])
    T_sup_right = np.array([-cos(yaw)*height,-sin(yaw)*height,height])
    T_inf_left = np.array([-cos(yaw)*height,-sin(yaw)*height,-height])
    T_ing_right = np.array([cos(yaw)*height,sin(yaw)*height,-height])
    vector_translation_corner = [T_sup_left,T_sup_right,T_inf_left,T_ing_right]

    # Translation matrix center
    T_front = np.array([cos(yaw+pi/4)*width,sin(yaw+pi/4)*width,0])
    T_back = np.array([cos(yaw-pi/4)*width,sin(yaw-pi/4)*width,0])
    vector_translation_center = [T_front,T_back]

    bounding_box = []
    for T_center in vector_translation_center:
        point_T_center = [point[0]+T_center[0],point[1]+T_center[1],point[2]+T_center[2],point[3],point[4],point[5],point[6]]
        for T_corner in vector_translation_corner:
            point_T_corner = [point_T_center[0]+T_corner[0],point_T_center[1]+T_corner[1],point_T_center[2]+T_corner[2],point_T_center[3],point_T_center[4],point_T_center[5],point_T_center[6]]
            bounding_box.append(point_T_corner)
    
    return bounding_box

    

def main():
    # Global variables
    config: dict[str, Any] = dict(
        link_name=None,
        link_pose=None,
        camera_matrix=None,
        img_rgb=None,
        bridge=None
    )

    # print(f'{Fore.RED}Hello world! {Style.RESET_ALL}')

    # Init Node
    rospy.init_node('label_data', anonymous=False)

    # Retrieving parameters
    image_raw_topic = rospy.get_param('~image_raw_topic', '/top_right_camera/image_raw')
    rate_hz = rospy.get_param('~rate', 30)
  
    # Subscribers
    PosesCallback_part = partial(poses_callback, config=config)
    rospy.Subscriber('gazebo/link_states', LinkStates, PosesCallback_part)
    ParameterCameraCallback_part = partial(parameter_camera_callback, config=config)
    rospy.Subscriber('/top_right_camera/camera_info', CameraInfo, ParameterCameraCallback_part)
    imgRgbCallback_part = partial(imgRgbCallback, config=config)
    rospy.Subscriber(image_raw_topic, Image, imgRgbCallback_part)
    

    # Create an object of the CvBridge class
    config['bridge'] = CvBridge()

    # set loop rate
    rate = rospy.Rate(rate_hz)
    listener = tf.TransformListener()

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

                elif name[1] == 'base_footprint':
                    camera_pose = {}
                    camera_pose['signal'] = name[0]
                    camera_pose['pose'] = config['link_pose'][i]
                    camera_poses.append(camera_pose)
            
            # Array of signals
            signal_pose_array = np.array([[pose['pose'].position.x,pose['pose'].position.y,pose['pose'].position.z,pose['pose'].orientation.x,pose['pose'].orientation.y,pose['pose'].orientation.z,pose['pose'].orientation.w] for pose in signal_poses],dtype = np.float64)
            signal_pose_xyz = np.array([[pose['pose'].position.x,pose['pose'].position.y,pose['pose'].position.z] for pose in signal_poses],dtype = np.float64)
            signal_name = np.array([pose['signal'] for pose in signal_poses])
           
            # Array of cameras
            base_footprint_pose_array = np.array([[pose['pose'].position.x,pose['pose'].position.y,pose['pose'].position.z,pose['pose'].orientation.x,pose['pose'].orientation.y,pose['pose'].orientation.z,pose['pose'].orientation.w] for pose in camera_poses],dtype = np.float64)
            
        
            bboxs = []
            for i in range(len(signal_pose_array)):
                bbox = get_bounding_box(signal_pose_array[i])
                bboxs.append(bbox)

            
            try:
                (trans,rot) = listener.lookupTransform('base_footprint', 'top_right_camera_rgb_optical_frame',  rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException):
                continue
            
            rot_matrix, trans_matrix = get_matrix(base_footprint_pose_array[0],trans,rot)

            points_2d = cv2.projectPoints(signal_pose_xyz, rot_matrix, trans_matrix, config['camera_matrix'], None,)[0]
            # homgeneous coordinates0
      
            # X = np.append(signal_pose_xyz[12],1)
            # RT = np.hstack((rot_matrix,np.array([np.array(x, ndmin=1) for x in trans_matrix])))
            # P = np.dot(config['camera_matrix'],RT)
            # x_2d = np.dot(P,X)
            # x_2d = x_2d/x_2d[-1]

            
            
            print('signal: ' + signal_name[12] + ' point center:' + str(points_2d[12]))


            # read opencv key
            win_name = 'Robot View'
            cv2.namedWindow(winname=win_name,flags=cv2.WINDOW_NORMAL)
            image = cv2.cvtColor(config['img_rgb'], cv2.COLOR_BGR2RGB)

            # print(image.shape)
            # 480,640,3
            cv2.imshow(win_name, image)
            key = cv2.waitKey(1)

            

        rate.sleep()

   
if __name__ == '__main__':
    main()