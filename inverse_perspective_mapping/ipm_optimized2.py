import time

import numpy as np
import cv2
import math
from sklearn import preprocessing
import datetime


class IPM():
    """
    inverse_perspective_mapping

    dimensions: meters, rad
    
    """

    def __init__(self, config_intrinsic, config_extrinsic):
        self.fov_x = config_intrinsic['fov_x']
        self.fov_y = config_intrinsic['fov_y']
        self.dim = config_intrinsic['img_dim']

        self.cam_height = config_extrinsic['camera_height']
        self.yaw = config_extrinsic['yaw']
        self.K = np.zeros([3, 3])

        self.calculate_intrinsic_matrix()
        self.calculate_extrinsic_matrix()
        self.calculate_global_matrix()
        self.calculate_A_const()

    def calculate_intrinsic_matrix(self):
        intrinsic_matrix = np.zeros([3, 3])

        # NO NEED WHEN USING GAZEBO CAMERA!
        # focal_length_x = (self.dim[0]/2) / (math.tan(self.fov_x/2)) # in pixels
        # focal_length_y = (self.dim[1]/2) / (math.tan(self.fov_y/2)) # in pixels
        # x0 = self.dim[0]/2
        # y0 = self.dim[1]/2

        # intrinsic_matrix[0,0] = focal_length_x
        # intrinsic_matrix[1,1] = focal_length_y
        # intrinsic_matrix[2,2] = 1
        # intrinsic_matrix[0,2] = x0
        # intrinsic_matrix[1,2] = y0

        # camera_info topic
        intrinsic_matrix[0, 0] = 563.62
        intrinsic_matrix[1, 1] = 563.62
        intrinsic_matrix[2, 2] = 1
        intrinsic_matrix[0, 2] = 340.5
        intrinsic_matrix[1, 2] = 240.5

        self.K = intrinsic_matrix

    def calculate_extrinsic_matrix(self):
        cRr = np.zeros([3, 3])
        cTr = np.zeros([3, 1])

        # cRr[0,0] = 1
        # cRr[1,1] = math.cos(self.yaw)
        # cRr[1,2] = -math.sin(self.yaw)
        # cRr[2,1] = math.sin(self.yaw)
        # cRr[2,2] = math.cos(self.yaw)

        cRr[0, 0] = math.cos(self.yaw)
        cRr[0, 2] = math.sin(self.yaw)
        cRr[1, 1] = 1
        cRr[2, 0] = -math.sin(self.yaw)
        cRr[2, 2] = math.cos(self.yaw)

        cTr[2] = self.cam_height

        self.P = np.matmul(self.K, cRr)
        self.t = np.matmul(self.K, cTr)

    def calculate_global_matrix(self):
        global_matrix = np.zeros([4, 4])
        global_matrix[0:3, 0:3] = self.P
        global_matrix[0:2, 3] = None
        global_matrix[2, 3] = -1
        global_matrix[3, 2] = 1

        self.A = global_matrix
        self.vector = np.zeros([4, 1])
        self.vector[0:3, 0] = -self.t[0:3, 0]

    def calculate_A_const(self):

        x_array = []
        y_array = []
        v_array = []

        index_point = 0
        for x in range(0, self.dim[0]):
            for y in range(0, self.dim[1]):
                self.A[0, 3] = -x
                self.A[1, 3] = -y

                (X, Y, _, __) = np.matmul(np.linalg.inv(self.A), self.vector)
                x_array.append(X)
                y_array.append(Y)

        minmax_scale_x = preprocessing.MinMaxScaler(feature_range=(0, self.dim[0] - 1))
        minmax_scale_y = preprocessing.MinMaxScaler(feature_range=(0, self.dim[1] - 1))

        x_array_scaled = minmax_scale_x.fit_transform(x_array).astype((int))
        y_array_scaled = minmax_scale_y.fit_transform(y_array).astype((int))
        self.x_array = x_array_scaled
        self.y_array = y_array_scaled

    def calculate_output_image(self, img_in):

        # Start the timer
        _t0 = time.time()

        v_array = np.array(img_in).ravel()
        output_image = np.zeros([self.dim[0], self.dim[1]])

        for i in range(0, len(self.x_array)):
            output_image[self.x_array[i], self.y_array[i]] = v_array[i]

        print(f"The image writing process takes: {time.time() - _t0}s")

        return output_image


def main():
    path = 'images/image1.png'
    img = cv2.imread(path, 2)  # gray image

    dim = (img.shape[0], img.shape[1])

    # no need config_intrinsic! only with real camera!
    config_intrinsic = {'fov_x': 1.09,
                        'fov_y': 1.09,
                        'img_dim': dim}

    config_extrinsic = {'camera_height': 0.547,
                        'yaw': 0.6}

    ipm = IPM(config_intrinsic, config_extrinsic)

    output_image = ipm.calculate_output_image(img)

    cv2.imshow('initial_image', img)
    cv2.imshow('final_image', output_image.astype(np.uint8))
    print(img.shape)
    print(output_image.dtype)
    print(img.dtype)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
