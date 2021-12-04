#!/usr/bin/env python3

import time
import numpy as np
import math
from sklearn import preprocessing


class IPM():
    """
        Class to do IPM (Inverse Perspective Mapping). The objetive is to get a BEV (Bird's Eye View) image
    """

    def __init__(self, width, height, K, R, cam_height):
        """
        Initiating the class, retrieving the variables, and calling needed functions
        """
        self.width = width
        self.height = height

        self.K = K
        self.R = R

        self.cam_height = cam_height

        self.calculate_extrinsic_matrix()
        self.calculate_global_matrix()
        self.calculate_A_const()


    def calculate_extrinsic_matrix(self):
        """
        From world parameters, calculate the extrinsic matrix
        """

        cTr = np.zeros([3, 1])
        cRr = self.R

        cTr[2] = self.cam_height

        self.P = np.matmul(self.K, cRr)
        self.t = np.matmul(self.K, cTr)

    def calculate_global_matrix(self):
        """
        From the extrinsic matrix, calculate the global matrix
        """

        global_matrix = np.zeros([4, 4])
        global_matrix[0:3, 0:3] = self.P
        global_matrix[0:2, 3] = None
        global_matrix[2, 3] = -1
        global_matrix[3, 2] = 1

        self.A = global_matrix
        self.vector = np.zeros([4, 1])
        self.vector[0:3, 0] = -self.t[0:3, 0]

    def calculate_A_const(self):
        """
        Calculates the A matrix
        """

        x_array = []
        y_array = []
        v_array = []

        index_point = 0
        for x in range(0, self.height):
            for y in range(0, self.width):
                self.A[0, 3] = -x
                self.A[1, 3] = -y

                (X, Y, _, __) = np.matmul(np.linalg.inv(self.A), self.vector)
                x_array.append(X)
                y_array.append(Y)

        minmax_scale_x = preprocessing.MinMaxScaler(feature_range=(0, self.height - 1))
        minmax_scale_y = preprocessing.MinMaxScaler(feature_range=(0, self.width - 1))

        x_array_scaled = minmax_scale_x.fit_transform(x_array).astype((int))
        y_array_scaled = minmax_scale_y.fit_transform(y_array).astype((int))
        self.x_array = tuple(x_array_scaled)
        self.y_array = tuple(y_array_scaled)

    def calculate_output_image(self, img_in):

        # Start the timer
        _t0 = time.time()

        v_array = np.array(img_in).ravel()
        output_image = np.zeros([self.height, self.width])

        for i in range(0, len(self.x_array)):
            output_image[self.x_array[i], self.y_array[i]] = v_array[i]

        print(f"The image writing process takes: {time.time() - _t0}s")

        return output_image