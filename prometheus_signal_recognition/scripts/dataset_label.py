#!/usr/bin/env python3

""""
This script is used to label the dataset for the signal recognition.
It will save the images and the labels in the folder specified in the config file.
"""

# Imports
import rospy
import cv2
import tf
import numpy as np
from typing import Any
from datetime import datetime
from functools import partial
from math import cos, sin, sqrt, pi
from scipy.spatial.transform import Rotation as R

from cv_bridge.core import CvBridge
from gazebo_msgs.msg._LinkStates import LinkStates
from sensor_msgs.msg._CameraInfo import CameraInfo
from sensor_msgs.msg._Image import Image

# Callback function to receive link states
def poses_callback(message, config: dict):
 
    config['link_name'] = message.name
    config['link_pose'] = message.pose


# Callback function to receive camera info
def parameter_camera_callback(message, config: dict):
    
    config['camera_matrix'] = message.K
    config['camera_matrix'] = np.reshape(config['camera_matrix'], (3, 3))
   
  
# Callback function to receive image
def imgRgbCallback(message, config: dict):
    
    config['img_rgb'] = config['bridge'].imgmsg_to_cv2(message, "rgb8")


# Convert quaternion to roll, pitch, yaw
def get_euler_from_quaternion(point_array):
    quaternion1 = (point_array[3],point_array[4],point_array[5],point_array[6])
    euler = tf.transformations.euler_from_quaternion(quaternion1, axes='sxyz') # will provide result in x, y,z sequence
    roll=euler[0]
    pitch=euler[1]
    yaw=euler[2]
    return roll, pitch, yaw


def angle_between_vectors(point_array1, point_array2):
    
    _,_,yaw1 = get_euler_from_quaternion(point_array1)
    _,_,yaw2 = get_euler_from_quaternion(point_array2)
    angle_between = yaw2 - (yaw1 - pi/2)
    if angle_between < 0:
        angle_between = angle_between + 2*pi
    
    return np.degrees(angle_between)

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
    point: [x,y,z]
    Returns:
    bounding_box: array with 4 corners of the bounding box
    """

    _, _, yaw = get_euler_from_quaternion(point)
    
    # Since the sign is square, the height is equal to the length.
    height = 0.305/2

    #Translation matrix corner 
    T_sup_left = np.array([cos(yaw)*height,sin(yaw)*height,height])
    T_sup_right = np.array([-cos(yaw)*height,-sin(yaw)*height,height])
    T_inf_left = np.array([-cos(yaw)*height,-sin(yaw)*height,-height])
    T_ing_right = np.array([cos(yaw)*height,sin(yaw)*height,-height])
    vector_translation_corner = [T_sup_left,T_sup_right,T_inf_left,T_ing_right]

    bounding_box = []
    for T_corner in vector_translation_corner:
        point_T_corner = [point[0]+T_corner[0],point[1]+T_corner[1],point[2]+T_corner[2]]
        bounding_box.append(point_T_corner)

    return bounding_box


def get_iou(bb1_2d,bb2_2d):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        bb1 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        """

        for i, bb_2d in enumerate([bb1_2d,bb2_2d]):
        
            min_x = min(bb_2d[0][0][0],bb_2d[1][0][0],bb_2d[2][0][0],bb_2d[3][0][0])
            max_x = max(bb_2d[0][0][0],bb_2d[1][0][0],bb_2d[2][0][0],bb_2d[3][0][0])
            min_y = min(bb_2d[0][0][1],bb_2d[1][0][1],bb_2d[2][0][1],bb_2d[3][0][1])
            max_y = max(bb_2d[0][0][1],bb_2d[1][0][1],bb_2d[2][0][1],bb_2d[3][0][1])
            if i == 0:
                bb1 = {'x1': min_x , 'x2': max_x , 'y1': min_y , 'y2': max_y}
            else:
                bb2 = {'x1': min_x , 'x2': max_x , 'y1': min_y , 'y2': max_y}
                
        
        # determine the coordinates of the intersection rectangle
        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])
        
        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = abs(max((x_right - x_left),0) * max((y_bottom - y_top),0))
        
        # compute the area of both AABBs
        bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])

        # compute the intersection over union 
        iou = intersection_area / float(bb1_area)

        return iou
    
def class_name2class_id(class_name):
    name = class_name.split('_')
    if name[0] == 'Bus':
        class_id = 1
    elif name[0] == 'Depression':
        class_id = 2
    elif name[0] == 'Turn':
        class_id  = 3
    elif name[0] == 'Road':
        class_id = 4    
    elif name[0] == 'Lights':
        class_id = 5
    elif name[0] == 'Hospital':
        class_id = 6
    elif name[0] == 'Park':
        class_id = 7
    elif name[0] == 'Crosswalk':
        class_id = 8
    elif name[0] == 'Other':
        class_id = 9
    elif name[0] == 'Round':
        class_id = 10
    elif name[0] == 'RMV':
        class_id = 11
    elif name[0] == 'Cattle':
        class_id = 12
    
    return class_id


def calculate_distance(point1, point2):
    """
    Calculates the distance between two points in 2D space.
    Returns a float representing the distance, with sign depending
    on the relative position of the points.
    """
    
    x1,y1 = point1[0],point1[1]
    x2,y2 = point2[0],point2[1]

    dx = x2 - x1
    dy = y2 - y1
    distance = sqrt(dx**2 + dy**2)
    
    if dx >= 0 and dy >= 0:
        # both x and y are positive
        return distance
    elif dx >= 0 and dy < 0:
        # x is positive, y is negative
        return -distance
    elif dx < 0 and dy >= 0:
        # x is negative, y is positive
        return distance
    else:
        # both x and y are negative
        return -distance


def main():
    ####################################
    # Init variables                   #
    ####################################
    # Global variables
    config: dict[str, Any] = dict(
        link_name=None,
        link_pose=None,
        camera_matrix=None,
        img_rgb=None,
        bridge=None
    )

    # Path to save the images and labels
    path_image = '/media/tatiana/E6D2-960C/AutoMec-AD/images/'
    path_label = '/media/tatiana/E6D2-960C/AutoMec-AD/labels/'
    path_image_p = '/media/tatiana/E6D2-960C/AutoMec-AD/images_p/'

    ####################################
    # ROS Inizialization               #
    ####################################
    # Init Node
    rospy.init_node('label_data', anonymous=False)

    # Retrieving parameters
    camera_name = rospy.get_param('~image_raw_topic', '/top_right_camera/image_raw')
    rate_hz = rospy.get_param('~rate', 30)
    
    # Subscribers
    PosesCallback_part = partial(poses_callback, config=config)
    rospy.Subscriber('/gazebo/link_states', LinkStates, PosesCallback_part)
    ParameterCameraCallback_part = partial(parameter_camera_callback, config=config)
    rospy.Subscriber('/top_right_camera/camera_info', CameraInfo, ParameterCameraCallback_part)
    imgRgbCallback_part = partial(imgRgbCallback, config=config)
    rospy.Subscriber(camera_name, Image, imgRgbCallback_part)
    
    # Create an object of the CvBridge class
    config['bridge'] = CvBridge()

    # set loop rate
    rate = rospy.Rate(rate_hz)
    listener = tf.TransformListener()

    while not rospy.is_shutdown():
        if not config['link_name'] is None and not config['link_pose'] is None and not config['camera_matrix'] is None:
            # init variables
            image_ori = cv2.cvtColor(config['img_rgb'], cv2.COLOR_BGR2RGB)
            signal_poses = []
            base_footprint = []

            ############################################
            # Get the signals and base_footprint       #
            ############################################
            for i in range(len(config['link_name'])):
                name = config['link_name'][i].split('::')

                if name[1] == 'pictogram':
                    signal_pose = {}
                    signal_pose['signal'] = name[0]
                    signal_pose['pose'] = config['link_pose'][i]
                    signal_poses.append(signal_pose)
                   
                elif name[1] == 'base_footprint':
                    base = {}
                    base['signal'] = name[0]
                    base['pose'] = config['link_pose'][i]
                    base_footprint.append(base)

            # Array of signals poses
            signal_pose_array = np.array([[pose['pose'].position.x,pose['pose'].position.y,pose['pose'].position.z,pose['pose'].orientation.x,pose['pose'].orientation.y,pose['pose'].orientation.z,pose['pose'].orientation.w] for pose in signal_poses],dtype = np.float64)
            signal_pose_xyz = np.array([[pose['pose'].position.x,pose['pose'].position.y,pose['pose'].position.z] for pose in signal_poses],dtype = np.float64)
            signal_name = np.array([pose['signal'] for pose in signal_poses])
           
            # Array of base_footprint poses
            base_footprint_pose_array = np.array([[pose['pose'].position.x,pose['pose'].position.y,pose['pose'].position.z,pose['pose'].orientation.x,pose['pose'].orientation.y,pose['pose'].orientation.z,pose['pose'].orientation.w] for pose in base_footprint],dtype = np.float64)

            ############################################
            #Transformation matrix: camera to world    #
            ############################################            
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
            
            ############################################
            # Get the bounding box of the signals      #
            ############################################
            bboxs = []
            for i in range(len(signal_pose_array)):
                bbox = get_bounding_box(signal_pose_array[i])
                bboxs.append(bbox)

            ############################################
            # Transform points 3d to 2d                #
            ############################################
            points_2d = cv2.projectPoints(signal_pose_xyz, cv2.Rodrigues(rot_matrix_cam2world)[0], trans_matrix_cam2world, config['camera_matrix'], None)[0].astype(np.int32)
            
            bbox_2d = []
            for corners in bboxs:
                bbox_2d.append(cv2.projectPoints(np.array(corners), cv2.Rodrigues(rot_matrix_cam2world)[0], trans_matrix_cam2world, config['camera_matrix'], None)[0].astype(np.int32))

            ############################################
            # Presence of the signal in the image      #
            ############################################
            image_with_point = config['img_rgb']
            images_with_signal = []

            points_test = signal_pose_xyz
            for idx_objects, _ in enumerate(points_test):
                point = np.ones((1,4))
                point[:,0:3] = points_test[idx_objects]
                point = np.transpose(point)
                point = np.dot(matrix_world2cam,point)
                points_test[idx_objects,:]=np.transpose(point[0:3,:])
            
            
            for i in range(len(points_2d)):
                # distance betwwen the camera and the signal
                dist = sqrt((base_footprint_pose_array[0][0]-signal_pose_array[i][0])**2 + (base_footprint_pose_array[0][1]-signal_pose_array[i][1])**2)
            
                # angle between the camera and the signal
                angle = angle_between_vectors(signal_pose_array[i],base_footprint_pose_array[0])
                
                # min and max point of the bounding box
                min_x = min(bbox_2d[i][0][0][0],bbox_2d[i][1][0][0],bbox_2d[i][2][0][0],bbox_2d[i][3][0][0])
                max_x = max(bbox_2d[i][0][0][0],bbox_2d[i][1][0][0],bbox_2d[i][2][0][0],bbox_2d[i][3][0][0])
                min_y = min(bbox_2d[i][0][0][1],bbox_2d[i][1][0][1],bbox_2d[i][2][0][1],bbox_2d[i][3][0][1])
                max_y = max(bbox_2d[i][0][0][1],bbox_2d[i][1][0][1],bbox_2d[i][2][0][1],bbox_2d[i][3][0][1])
                
                # area of the bounding box
                area = (max_x-min_x)*(max_y-min_y)
                if round(area/500) > 100:
                    tresshold_pose = 100
                else:
                    tresshold_pose = round(area/500)
               
                # if i == 1:
                #     print((max_x-min_x)/image_ori.shape[1])

                # If the signal is in the image and the signal is in front of the camera:
                if min_x > -tresshold_pose and max_x < (image_ori.shape[1]+tresshold_pose) and min_y > -tresshold_pose and max_y < (image_ori.shape[0]+tresshold_pose) and points_test[i][2]>0 and angle > 100 and angle < 260 and area > 300 and (max_x-min_x)/image_ori.shape[1] > 0.02:
                    
                    # Dictionary of signal
                    signal = {}
                    signal['name'] = signal_name[i]
                    signal['bbox'] = bbox_2d[i]
                    signal['x_center'] = points_2d[i][0][0]
                    signal['y_center'] = points_2d[i][0][1]
                    signal['distance'] = dist

                    if images_with_signal == []:
                        images_with_signal.append(signal)
                        image_with_point = cv2.circle(config['img_rgb'], (signal['x_center'],signal['y_center']), 4, (0, 255, 255), 2)
                       
                    else:
                        iou_TRUE = False
                        for signal_image in images_with_signal:
                            if signal_image['name'] == signal['name']:
                                continue
                            
                            iou = get_iou(signal_image['bbox'],signal['bbox'])
                            if iou != 0 and (iou != 1 or (iou == 1 and signal_image['name'] != signal['name'])) :
                                if signal_image['distance'] > signal['distance']:
                                    images_with_signal.remove(signal_image)
                                    image_with_point = cv2.circle(config['img_rgb'], (signal_image['x_center'],signal_image['y_center']), 4, (255, 0, 0), 3)
                                    continue
                                else:
                                    iou_TRUE = True

                        if iou_TRUE == False:
                            images_with_signal.append(signal)
                            image_with_point = cv2.circle(config['img_rgb'], (signal['x_center'],signal['y_center']), 4, (0, 255, 255), 2)
                           
              
            ############################################
            # Visualize the image                      #
            ############################################
            win_name = 'Robot View'
            cv2.namedWindow(winname=win_name,flags=cv2.WINDOW_NORMAL)
            image = cv2.cvtColor(image_with_point, cv2.COLOR_BGR2RGB)
            cv2.imshow(win_name, image)

            ############################################
            # Save the image and the label             #
            ############################################
            if len(images_with_signal) > 0:
                # Save the images
                timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                cv2.imwrite(f'{path_image}signal{timestamp}.jpg',image_ori)
                cv2.imwrite(f'{path_image_p}signal{timestamp}.jpg',image)
                
                # Create a file to save the labels
                with open(f'{path_label}signal{timestamp}.txt', "w") as f:
                    
                    # write the values to the file
                    for i,signal in enumerate(images_with_signal):
                        class_signal = class_name2class_id(signal['name'])
                        x_center_norm = signal['x_center']/image_ori.shape[1]
                        y_center_norm = signal['y_center']/image_ori.shape[0]                       
                        width_norm = (max(signal['bbox'][0][0][0],signal['bbox'][1][0][0],signal['bbox'][2][0][0],signal['bbox'][3][0][0])-min(signal['bbox'][0][0][0],signal['bbox'][1][0][0],signal['bbox'][2][0][0],signal['bbox'][3][0][0]))/image_ori.shape[1]
                        height_norm = (max(signal['bbox'][0][0][1],signal['bbox'][1][0][1],signal['bbox'][2][0][1],signal['bbox'][3][0][1])-min(signal['bbox'][0][0][1],signal['bbox'][1][0][1],signal['bbox'][2][0][1],signal['bbox'][3][0][1]))/image_ori.shape[0]

                        f.write(f'{class_signal} {x_center_norm} {y_center_norm} {width_norm} {height_norm}')
                        f.write('\n')
                    f.close()

            key = cv2.waitKey(1)     

        rate.sleep()

   
if __name__ == '__main__':
    main()