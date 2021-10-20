#!/usr/bin/env python3
#
# bird's eye view calculated from transform matrix computation 
# from intrinsic and extrinsic camera parameters and 
# using opencv WrapPerspective method 
# 
# The transformation is applied after image undistort procedure 
#
# v1.0.1 On 2021/10/02 by inaciose
#

import numpy as np
import cv2
import math

class bev():

    def __init__(self, cfg_distortion, cfg_intrinsic, cfg_extrinsic):

        self.PI = 3.1415926

        # set camera rotation from ground referencial
        # get from normal and convert to radians
        # to calculate the rotation matrix
        self.pitch = (cfg_extrinsic['pitch'] - 90) * self.PI/180
        self.roll = (cfg_extrinsic['roll'] - 90) * self.PI/180
        self.yaw = (cfg_extrinsic['yaw'] - 90) * self.PI/180

        # camera translation from ground referencial (origin) mm
        self.tx = cfg_extrinsic['tx']   # lateral distance to camera origin 
        self.ty = cfg_extrinsic['ty']   # forward distance to camera origin
        self.tz = cfg_extrinsic['tz']   # height distance to camera origin // 500 // 238

        # image size, pixels
        self.img_width = cfg_intrinsic['iw']
        self.img_height = cfg_intrinsic['ih']

        # camera matrix
        self.focal_lenght_x = cfg_intrinsic['fx'] # 500 // 559
        self.focal_lenght_y = cfg_intrinsic['fy'] # 500 // 559
        self.skew = cfg_intrinsic['sk']
        self.optical_center_x = cfg_intrinsic['cx']
        self.optical_center_y = cfg_intrinsic['cy']
        
        # initialize main matrices (only for reference)
        self.D = np.zeros([1, 5])
        self.P = np.zeros([4, 3])
        self.R = np.zeros([4, 4])
        self.T = np.zeros([4, 4])
        self.K = np.zeros([3, 4])
        self.TM = np.zeros([4, 4])

        # initialize undistort matrices (only for reference)
        self.CM = np.zeros([3, 3]) 
        # from getOptimalNewCameraMatrix()
        self.NCM = None                   
        self.undistort_roi = None
        # from initUndistortRectifyMap
        self.img_mapx = None
        self.img_mapy = None

        # set main matrices
        self.set_distortion_matrix(cfg_distortion['k1'], cfg_distortion['k2'], cfg_distortion['p1'], cfg_distortion['p2'], cfg_distortion['k3'])
        self.set_projection_matrix()
        self.set_rotation_matrix()
        self.set_translation_matrix()
        self.set_camera_matrix()
        self.set_transformation_matrix()

        # set undistort matrices
        self.set_optimal_new_camera_matrix()
        self.set_undistort_rectify_map()

    def set_distortion_matrix(self, k1, k2, p1, p2, k3):
        self.D = np.array([k1, k2, p1, p2, k3])

    def set_transformation_matrix(self):
        # self.TM = K * (T * (R * P));
        matrix1 = np.zeros([4, 4])
        matrix2 = np.zeros([4, 4])
        
        matrix1 = np.matmul(self.R, self.P)
        matrix2 = np.matmul(self.T, matrix1)
        
        self.TM = np.matmul(self.K, matrix2)

    def set_projection_matrix(self):
        matrix = np.zeros([4, 3])
        matrix[0, 0] = 1
        matrix[0, 2] = -self.img_width/2
        matrix[1, 1] = 1
        matrix[1, 2] = -self.img_height/2
        matrix[3, 2] = 1

        self.P = matrix

    def set_rotation_matrix(self):
        # Rx pitch
        pmatrix = np.zeros([4, 4])
        pmatrix[0, 0] = 1
        pmatrix[1, 1] = math.cos(self.pitch)
        pmatrix[1, 2] = -math.sin(self.pitch)
        pmatrix[2, 1] = math.sin(self.pitch)
        pmatrix[2, 2] = math.cos(self.pitch)
        pmatrix[3, 3] = 1
        # Ry roll
        rmatrix = np.zeros([4, 4])
        rmatrix[0, 0] = math.cos(self.roll)
        rmatrix[0, 2] = -math.sin(self.roll)
        rmatrix[1, 1] = 1
        rmatrix[2, 0] = math.sin(self.roll)
        rmatrix[2, 2] = math.cos(self.roll)
        rmatrix[3, 3] = 1
        # Rz yaw
        ymatrix = np.zeros([4, 4])
        ymatrix[0, 0] = math.cos(self.yaw)
        ymatrix[0, 1] = -math.sin(self.yaw)
        ymatrix[1, 0] = math.sin(self.yaw)
        ymatrix[1, 1] = math.cos(self.yaw)
        ymatrix[2, 2] = 1
        ymatrix[3, 3] = 1
        #self.R = pmatrix * rmatrix * ymatrix
        matrix = np.zeros([4, 4])
        matrix = np.matmul(rmatrix, ymatrix)
        self.R = np.matmul(pmatrix, matrix)

    def set_translation_matrix(self):
        matrix = np.zeros([4, 4])
        matrix[0, 0] = 1
        matrix[0, 3] = self.tx
        matrix[1, 1] = 1
        matrix[1, 3] = self.ty
        matrix[2, 2] = 1
        matrix[2, 3] = self.tz
        matrix[3, 3] = 1

        self.T = matrix   

    def set_camera_matrix(self):
        # fx, fy, sk, cx, cy
        # camera_matrix with one more column
        # used to compute transformation matrix
        matrix = np.zeros([3, 4])
        matrix[0, 0] = self.focal_lenght_x
        matrix[0, 1] = self.skew
        matrix[0, 2] = self.optical_center_x
        matrix[1, 1] = self.focal_lenght_y
        matrix[1, 2] = self.optical_center_y
        matrix[2, 2] = 1

        self.K = matrix

        # same matrix, without last column
        # used to undistort image
        matrix = np.zeros([3, 3])
        matrix[0, 0] = self.focal_lenght_x
        matrix[0, 1] = self.skew
        matrix[0, 2] = self.optical_center_x
        matrix[1, 1] = self.focal_lenght_y
        matrix[1, 2] = self.optical_center_y
        matrix[2, 2] = 1

        self.CM = matrix

    # undistort preparation (roi contains the coords for the rectangle without black pixels from undistort transformation)
    def set_optimal_new_camera_matrix(self):
        nmc, roi = cv2.getOptimalNewCameraMatrix(self.CM, self.D, (self.img_width, self.img_height), 1, (self.img_width, self.img_height))
        self.NCM = nmc                   
        self.undistort_roi = roi

    # undistort preparation (required only for alternative 2: quick way)
    def set_undistort_rectify_map(self):
        mapx, mapy = cv2.initUndistortRectifyMap(self.CM, self.D, None, self.NCM, (self.img_width, self.img_height), 5)
        self.img_mapx = mapx
        self.img_mapy = mapy

    # undistort (alternative 1: easy way)
    def get_undistorted_image(self, img):
        return cv2.undistort(img, self.CM, self.D, None, self.NCM)

    # undistort (alternative 2: quick way)
    def get_remaped_image(self, img):
        return cv2.remap(img, self.img_mapx, self.img_mapy, cv2.INTER_LINEAR)

    # crop undistorted image to the roi obtained
    # in getOptimalNewCameraMatrix
    def get_croped_image_from_undistort_roi(self):
        x, y, w, h = self.undistort_roi
        return undistorted_image[y:y+h, x:x+w]

    def get_bev_image(self, img):
        return cv2.warpPerspective(img, self.TM, img.shape[:2][::-1], flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)


def main():

    # minimal recomended reading for the configuration
    # for distortion, intrinsic and extrinsic parameters of the camera read
    # https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
    
    # radial and tangential distortion coefficients
    cfg_distortion = {   'k1': 0.001593932351222154,
                            'k2': -0.004430239769797794,
                            'p1': 0.0002036043769375994,
                            'p2': -3.141393111217492e-06,
                            'k3': 0}

    # Intrinsic parameters are specific to a camera can be obtained by the 
    # the calibration procedure, or by calculations (all parameters in pixels)
    # added image width and height (iw, ih), in pixels, as extra parameters
    # focal length (fx,fy) pixels, skew (in pinhole model is usualy 0) 
    # and optical center of the image plane (cx,cy) pixels
    # alternatively we may use the folowing calculations
    # focal_length_ = w / (2 * math.tan(hfov/2)), in pixels
    # cx = image_width / 2, cy = image_height / 2 

    cfg_intrinsic = {    'fx': 564.1794158410564,
                            'fy': 563.8954010066177,
                            'sk': 0,                        
                            'cx': 339.585354927923,
                            'cy': 240.8593643042658,
                            'iw': 680,
                            'ih': 480}

    # Extrinsic parameters corresponds to rotation and translation vectors 
    # which translates a coordinates of a 3D point to a coordinate system
    # translations from world (tx, ty, tz) mm, 
    # rotations from world (pitch, roll, yaw) degrees
    cfg_extrinsic = {    'tx': 0,
                            'ty': 0,                        
                            'tz': 547,                        
                            'pitch': 34.4,                        
                            'roll': 90,                        
                            'yaw': 90}

    # load image and start processing
    path = 'imageL.png'
    img = cv2.imread(path, 1)  # 0 = gray, 1 = color
    #img = cv2.resize(img, (640,480))

    # create ipm class instance
    ipm = bev(cfg_distortion, cfg_intrinsic, cfg_extrinsic)

    #undistorted_image = ipm.get_undistorted_image(img)
    undistorted_image = ipm.get_remaped_image(img)
    ipm_img = ipm.get_bev_image(undistorted_image)

    # show result
    cv2.imshow('final_image', ipm_img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()