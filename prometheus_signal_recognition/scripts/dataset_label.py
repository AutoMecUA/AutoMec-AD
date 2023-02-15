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
    # config['camera_matrix'][0][2] = 320
    # config['camera_matrix'][1][2] = 240
  

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


# Get the transformation matrix from the point array
def get_matrix_from_PointArray(point_array):

    # rotation matrix
    r = R.from_quat([point_array[3], point_array[4], point_array[5], point_array[6]])
    rot_matrix = r.as_matrix()
    
    # translation matrix
    trans_matrix = np.array([point_array[0], point_array[1], point_array[2]], dtype=np.float32)

    # homogeneous transformation matrix (4,4) 
    l = 4
    matrix = np.zeros((l,l))
    matrix[0:3,0:3] = rot_matrix
    matrix[0:3,3] = trans_matrix
    matrix[3,3] = 1
      
    return matrix


# Get the transformation matrix from the translation and rotation
def get_matrix_from_TransRot(trans,rot):

    r = R.from_quat(rot)
    rot = r.as_matrix()

    # homogeneous transformation matrix (4,4) 
    l = 4
    matrix = np.zeros((l,l))
    matrix[0:3,0:3] = rot
    matrix[0:3,3] = trans
    matrix[3,3] = 1

    return matrix


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


    # Init Node
    rospy.init_node('label_data', anonymous=False)

    # Retrieving parameters
    image_raw_topic = rospy.get_param('~image_raw_topic', '/top_right_camera/image_raw')
    rate_hz = rospy.get_param('~rate', 30)
  
    # Subscribers
    PosesCallback_part = partial(poses_callback, config=config)
    rospy.Subscriber('/gazebo/link_states', LinkStates, PosesCallback_part)
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
        if not config['link_name'] is None and not config['link_pose'] is None and not config['camera_matrix'] is None:
            signal_poses = []
            camera_poses = []
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

            # Gather transformation from base_footprint to the camera 
            try:
                (trans_footprint2cam,rot_footprint2cam) = listener.lookupTransform('top_right_camera_rgb_optical_frame', 'base_footprint', rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException):
                continue
            matrix_footprint2cam = get_matrix_from_TransRot(trans_footprint2cam,rot_footprint2cam)

            # Gather transformation from world to base_footprint
            matrix_world2footprint = np.linalg.inv(get_matrix_from_PointArray(base_footprint_pose_array[0]))

            # Gather transformation from world to the camera
            matrix_world2cam = np.dot(matrix_footprint2cam, matrix_world2footprint)
            rot_matrix_cam2world = matrix_world2cam[0:3,0:3]
            trans_matrix_cam2world = matrix_world2cam[0:3,3]

            # Gather points in camera 2d coordinate frame
            points_2d = cv2.projectPoints(signal_pose_xyz, cv2.Rodrigues(rot_matrix_cam2world)[0], trans_matrix_cam2world, config['camera_matrix'], None)[0].astype(np.int32)

            # Display first signal with marking
            image_with_point = cv2.circle(config['img_rgb'], tuple(points_2d[12][0]), 4, (0, 255, 255), 2)

            # read opencv key
            win_name = 'Robot View'
            cv2.namedWindow(winname=win_name,flags=cv2.WINDOW_NORMAL)
            image = cv2.cvtColor(image_with_point, cv2.COLOR_BGR2RGB)

            cv2.imshow(win_name, image)
            key = cv2.waitKey(1)     

        rate.sleep()

   
if __name__ == '__main__':
    main()