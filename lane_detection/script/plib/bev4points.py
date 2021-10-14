#!/usr/bin/env python3
#
# from lib.bev4points import bev
# v1.0.0 by inaciose
#

import cv2
import numpy as np

class bev():
    def __init__(self, cfg_img, cfg_d, cfg_k, src_points, dst_points ):
        # not on args
        # cfg_img, cfg_d, cfg_k need to be passed as named arrays (sample)
        # cfg_img = { 'sw': 640, 'sh': 480}
        # points to calculate transformation need to be passed as numpy array (sample)
        # src_points = np.array([[196, 217], [441, 217], [517, 423], [120, 423]], dtype=np.float32)

        # image sizde
        self.src_width = cfg_img['sw']
        self.src_height = cfg_img['sh']

        # distortion matrix
        self.D = self.get_distortion_matrix(cfg_d['k1'], cfg_d['k2'], cfg_d['p1'], cfg_d['p2'], cfg_d['k3'])

        # intrinsic matrix
        self.K = self.get_intrinsic_matrix( cfg_k['fx'], cfg_k['fy'], cfg_k['sk'], cfg_k['cx'], cfg_k['cy'])

        # prepare undistort
        self.set_optimal_new_camera_matrix()
        self.set_undistort_rectify_map()

        # set matrix for bev transform
        self.transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    def get_distortion_matrix(self, k1, k2, p1, p2, k3):
        return np.array([k1, k2, p1, p2, k3])

    def get_intrinsic_matrix(self, focal_lenght_x, focal_lenght_y, skew, optical_center_x, optical_center_y):
        # fx, fy, sk, cx, cy
        matrix = np.zeros([3, 3])
        matrix[0, 0] = focal_lenght_x
        matrix[0, 1] = skew
        matrix[0, 2] = optical_center_x
        matrix[1, 1] = focal_lenght_y
        matrix[1, 2] = optical_center_y
        matrix[2, 2] = 1
        return matrix

    # undistort preparation (roi contains the coords for the rectangle without black pixels from undistort transformation)
    def set_optimal_new_camera_matrix(self):
        nmc, roi = cv2.getOptimalNewCameraMatrix(self.K, self.D, (self.src_width, self.src_height), 1, (self.src_width, self.src_height))
        self.NCM = nmc                   
        self.undistort_roi = roi

    # undistort preparation (required only for alternative 2: quick way)
    def set_undistort_rectify_map(self):
        mapx, mapy = cv2.initUndistortRectifyMap(self.K, self.D, None, self.NCM, (self.src_width, self.src_height), 5)
        self.img_mapx = mapx
        self.img_mapy = mapy

    # undistort (alternative 1: easy way)
    def get_undistorted_image(self, img):
        return cv2.undistort(img, self.CM, self.D, None, self.NCM)

    # undistort (alternative 2: quick way)
    def get_remaped_image(self, img):
        return cv2.remap(img, self.img_mapx, self.img_mapy, cv2.INTER_LINEAR)

    def getWarpPerspective(self, img):
        return cv2.warpPerspective(img, self.transform_matrix, img.shape[:2][::-1])

