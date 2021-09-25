import numpy as np
import cv2
import math
from sklearn import preprocessing
import datetime

class IPM():
    def __init__(self, config_intrinsic, config_extrinsic):
        self.fov_x = config_intrinsic['fov_x']
        self.fov_y = config_intrinsic['fov_y']
        self.dim = config_intrinsic['img_dim']

        self.cam_height = config_extrinsic['camera_height']
        self.yaw = config_extrinsic['yaw']

        self.calculate_intrinsic_matrix()
        self.calculate_extrinsic_matrix()
        self.calculate_global_matrix()

    def calculate_intrinsic_matrix(self):
        intrinsic_matrix = np.zeros([3,3])
        focal_length_x = (self.dim[0]/2) / (math.tan(self.fov_x/2)) # in pixels
        focal_length_y = (self.dim[1]/2) / (math.tan(self.fov_y/2)) # in pixels
        x0 = self.dim[0]/2
        y0 = self.dim[1]/2

        # intrinsic_matrix[0,0] = focal_length_x
        # intrinsic_matrix[1,1] = focal_length_y
        # intrinsic_matrix[2,2] = 1
        # intrinsic_matrix[0,2] = x0
        # intrinsic_matrix[1,2] = y0

        # camera_info topic
        intrinsic_matrix[0,0] = 563.62
        intrinsic_matrix[1,1] = 563.62
        intrinsic_matrix[2,2] = 1
        intrinsic_matrix[0,2] = 340.5
        intrinsic_matrix[1,2] = 240.5     

        self.K = intrinsic_matrix


    def calculate_extrinsic_matrix(self):
        cRr = np.zeros([3,3])
        cTr = np.zeros([3,1])

        # cRr[0,0] = 1
        # cRr[1,1] = math.cos(self.yaw)
        # cRr[1,2] = -math.sin(self.yaw)
        # cRr[2,1] = math.sin(self.yaw)
        # cRr[2,2] = math.cos(self.yaw)

        cRr[0,0] = math.cos(self.yaw)
        cRr[0,2] = math.sin(self.yaw)
        cRr[1,1] = 1
        cRr[2,0] = -math.sin(self.yaw)
        cRr[2,2] = math.cos(self.yaw)
        

        cTr[2] = self.cam_height

        self.P = np.matmul(self.K,cRr)
        self.t = np.matmul(self.K,cTr)

    def calculate_global_matrix(self):
        global_matrix = np.zeros([4,4])
        global_matrix[0:3,0:3] = self.P
        global_matrix[0:2,3] = None
        global_matrix[2,3] = -1
        global_matrix[3,2] = 1

        self.A = global_matrix
        self.vector = np.zeros([4,1])
        self.vector[0:3,0] = -self.t[0:3,0]


    def calculate_new_points_list(self, img_in):

        new_points_list = []

        for x in range(0,self.dim[0]):
            for y in range(0,self.dim[1]):
                self.A[0,3] = -x
                self.A[1,3] = -y

                (X,Y,_,__) = np.matmul(np.linalg.inv(self.A),self.vector)

                new_points_list.append(new_point(X,Y,img_in[x,y]))

        self.points_list = new_points_list

        return self.calculate_output_image()
    
    def calculate_output_image(self):

        x_array = []
        y_array = []
        v_array = []
        
        for i in range(0,len(self.points_list)):
            x_array.append(self.points_list[i].x)
            y_array.append(self.points_list[i].y)
            v_array.append(self.points_list[i].value)
        
        x_array = np.array(x_array)
        y_array = np.array(y_array)
        v_array = np.array(v_array).astype(np.uint8)

        print('X minimum : ' + str(x_array.min()))
        print('X maximum : ' + str(x_array.max()))
        print('Y minimum : ' + str(y_array.min()))
        print('Y maximum : ' + str(y_array.max()))

        minmax_scale_x = preprocessing.MinMaxScaler(feature_range=(0, self.dim[0]-1))
        minmax_scale_y = preprocessing.MinMaxScaler(feature_range=(0, self.dim[1]-1))

        x_array_scaled = minmax_scale_x.fit_transform(x_array).astype((int))
        y_array_scaled = minmax_scale_y.fit_transform(y_array).astype((int))

        print('X minimum : ' + str(x_array_scaled.min()))
        print('X maximum : ' + str(x_array_scaled.max()))
        print('Y minimum : ' + str(y_array_scaled.min()))
        print('Y maximum : ' + str(y_array_scaled.max()))

        output_image = np.zeros([self.dim[0],self.dim[1]])

        for i in range(0, len(x_array_scaled)):
            output_image[x_array_scaled[i],y_array_scaled[i]] = v_array[i]
            
        return output_image



class new_point():
    def __init__(self,x,y,value):
        self.x = x
        self.y = y
        self.value = value
        
        
def main():
    
  
    path = 'images/image2.png'
    img = cv2.imread(path,2) #gray image


    dim = (img.shape[0], img.shape[1])

    config_intrinsic = {'fov_x' : 1.09,
                        'fov_y' : 1.09,
                        'img_dim' : dim}

    config_extrinsic = {'camera_height' : 0.547,
                        'yaw' : 0.6 }
    
    ipm = IPM(config_intrinsic,config_extrinsic)

    output_image = ipm.calculate_new_points_list(img)


    cv2.imshow('initial_image', img)
    cv2.imshow('final_image', output_image.astype(np.uint8))
    print(output_image.dtype)
    print(img.dtype)
    cv2.waitKey(0)
    




if __name__ == "__main__":
    main()