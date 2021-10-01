#
# sergio.inacio@ua.pt
#
import numpy as np
import cv2
import math

class IPM():
    """
    dimensions: milimeters, grades
    """

    def __init__(self):

        PI = 3.1415926

        self.img_width = 680
        self.img_height = 480

        alpha = 35
        beta = 90
        gama = 90

        # convert to radians
        # to calculate the rotation matrix
        self.pitch = (alpha - 90) * PI/180
        self.roll = (beta - 90) * PI/180
        self.yaw = (gama - 90) * PI/180

        # camera translation from ground referencial (origin)
        self.tx = 0     # lateral distance to camera origin 
        self.ty = 0     # forward distance to camera origin
        self.tz = 547   # height distance to camera origin // 500 // 238

        self.focal_lenght_x = 563.62 # 500 // 559
        self.focal_lenght_y = 563.62 # 500 // 559
        self.skew = 0

        self.P = np.zeros([4, 3])
        self.R = np.zeros([4, 4])
        self.T = np.zeros([4, 4])
        self.K = np.zeros([3, 4])

        self.TM = np.zeros([4, 4])
        
        self.calculate_projection_matrix(self.img_width, self.img_height)
        self.calculate_rotation_matrix(self.pitch, self.roll, self.yaw)
        self.calculate_translation_matrix(self.tx, self.ty, self.tz)
        self.calculate_intrinsic_matrix(self.focal_lenght_x, self.focal_lenght_y, self.skew, self.img_width, self.img_height)

        self.calculate_transformation_matrix()

    def calculate_transformation_matrix(self):
        matrix1 = np.zeros([4, 4])
        matrix2 = np.zeros([4, 4])
        
        #self.TM = K * (T * (R * P));
        matrix1 = np.matmul(self.R, self.P)
        matrix2 = np.matmul(self.T, matrix1)

        self.TM = np.matmul(self.K, matrix2)

    def calculate_projection_matrix(self, width, height):
        matrix = np.zeros([4, 3])
        matrix[0, 0] = 1
        matrix[0, 2] = -width/2
        matrix[1, 1] = 1
        matrix[1, 2] = -height/2
        matrix[3, 2] = 1

        self.P = matrix

    def calculate_rotation_matrix(self, pitch, roll, yaw):

        # Rx pitch
        pmatrix = np.zeros([4, 4])
        pmatrix[0, 0] = 1
        pmatrix[1, 1] = math.cos(pitch)
        pmatrix[1, 2] = -math.sin(pitch)
        pmatrix[2, 1] = math.sin(pitch)
        pmatrix[2, 2] = math.cos(pitch)
        pmatrix[3, 3] = 1

        # Ry roll
        rmatrix = np.zeros([4, 4])
        rmatrix[0, 0] = math.cos(roll)
        rmatrix[0, 2] = -math.sin(roll)
        rmatrix[1, 1] = 1
        rmatrix[2, 0] = math.sin(roll)
        rmatrix[2, 2] = math.cos(roll)
        rmatrix[3, 3] = 1

        # Rz yaw
        ymatrix = np.zeros([4, 4])
        ymatrix[0, 0] = math.cos(yaw)
        ymatrix[0, 1] = -math.sin(yaw)
        ymatrix[1, 0] = math.sin(yaw)
        ymatrix[1, 1] = math.cos(yaw)
        ymatrix[2, 2] = 1
        ymatrix[3, 3] = 1

        #self.R = pmatrix * rmatrix * ymatrix
        matrix = np.zeros([4, 4])
        matrix = np.matmul(rmatrix, ymatrix)
        self.R = np.matmul(pmatrix, matrix)

    def calculate_translation_matrix(self, tx, ty, tz):
        matrix = np.zeros([4, 4])
        matrix[0, 0] = 1
        matrix[0, 3] = tx
        matrix[1, 1] = 1
        matrix[1, 3] = ty
        matrix[2, 2] = 1
        matrix[2, 3] = tz
        matrix[3, 3] = 1

        self.T = matrix   

    def calculate_intrinsic_matrix(self, focal_lenght_x, focal_lenght_y, skew, width, height):
        matrix = np.zeros([3, 4])
        matrix[0, 0] = focal_lenght_x
        matrix[0, 1] = skew
        matrix[0, 2] = width/2
        matrix[1, 1] = focal_lenght_y
        matrix[1, 2] = height/2
        matrix[2, 2] = 1

        self.K = matrix

def main():
    path = 'imageL.png'
    img = cv2.imread(path, 2)  # gray image
    img = cv2.resize(img, (640,480))

    ipm = IPM()

    print(ipm.TM)

    ipm_img = cv2.warpPerspective(img, ipm.TM, img.shape[:2][::-1], flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)

    cv2.imshow('initial_image', img)
    cv2.imshow('final_image', ipm_img.astype(np.uint8))
    print(img.shape)
    print(ipm_img.dtype)
    print(ipm_img.dtype)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
    
