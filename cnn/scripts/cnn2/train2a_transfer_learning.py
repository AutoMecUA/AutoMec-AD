#!/usr/bin/env python3

import csv
import cv2
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.optimizers import Adam 

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import rospy
import pathlib

def createModel():
    model = Sequential()
    model.add(Convolution2D(24, (5,5),(2, 2), input_shape = (160,320,1), activation='elu'))
    model.add(Convolution2D(36,(5,5),(2, 2), activation='elu'))
    model.add(Convolution2D(48, (5,5),(2, 2), activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))

    #model.add(Dropout(0.8))
    model.add(Flatten())
    model.add(Dense(100,activation='elu'))
    model.add(Dense(50,activation='elu'))
    model.add(Dense(10,activation='elu'))
    model.add(Dense(1))
    return model


def main():

    # Init Node
    rospy.init_node('ml_training', anonymous=False)

    base_folder = rospy.get_param('~base_folder', 'set1') 
    modelname = rospy.get_param('~modelname', 'model_default.h5')
    nb_epoch = rospy.get_param('~epochs', 20)
    batch_size = rospy.get_param('~batch_size', 32)

    rospy.loginfo('base_folder: %s', base_folder)
    rospy.loginfo('modelname: %s', modelname)
    rospy.loginfo('epochs: %s', nb_epoch)
    rospy.loginfo('batch_size: %s', batch_size)

    # set path
    s = str(pathlib.Path(__file__).parent.absolute())
    path_model = s + '/../../models/'
    path_data = s + '/../../data/' + base_folder
    image_path = path_data + '/IMG/'

    rospy.loginfo('Full path:\n %s', path_data)

    total_left_angles = 0
    total_right_angles = 0
    total_straight_angles = 0

    reader_lines = open(path_data + '/driving_log.csv')
    number_of_rows = len(reader_lines.readlines())

    

    n_times_tf = 10

    intervals = [xx for xx in range(0,number_of_rows,int(number_of_rows/n_times_tf))]




    for cycle in range(len(intervals)-1):
        limits = [intervals[cycle],intervals[cycle+1]]
        images = []
        angles = []


        with open(path_data + '/driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            k = 0
            for line in reader:
                k+=1
                if k<limits[0] or k>limits[1]:
                    continue


                center_image = cv2.imread(image_path + line[0],0)
                center_image = cv2.resize(center_image, (320, 160))
                center_image_temp = np.expand_dims(center_image, axis=2)

                images.append(center_image_temp)

                angles.append(round(float(line[1]),3))
                
                # Change image and store the same angle, with some distortion, both in image and in angle (to increase robustness)
                # flipped
                center_image_temp = np.expand_dims(cv2.flip(center_image, 1), axis=2)
                
                # images.append(center_image_temp)
                # angles.append(-float(line[1]))
                # no more distortions
                
                # do some stats
                if(float(line[1]) < -0.1):
                    total_left_angles += 1
                elif(float(line[1]) > 0.1):
                    total_right_angles += 1
                else:
                    total_straight_angles += 1
        
        csvfile.close()


        left_to_straight_ratio = total_straight_angles/total_left_angles
        right_to_straight_ratio = total_straight_angles/total_right_angles

        #print('angles are: ' + str(list(angles)))

        print('Total Samples : ', len(images))
        print()
        print('Initial Angle Distribution')
        print('Total Left Angles : ', total_left_angles)
        print('Total Right Angles : ', total_right_angles)
        print('Total Straight Angles : ', total_straight_angles)
        print('Left to Straight Ratio : ', left_to_straight_ratio)
        print('Right to Straight Ratio : ', right_to_straight_ratio)


        images = np.array(images,dtype=np.float16)

        images /= 255
        
        angles = np.array(angles, dtype=np.float16)


        #X_train, X_test, y_train, y_test = train_test_split(images, angles, test_size=0.00001)


        X_train = images
        y_train = angles

        X_test= []
        y_test = []




        path = path_model + 'cnn2a_tf_' + modelname


        try:
            model = load_model(path)
        except:
            model = createModel()

    


        model.summary()

        model.compile(Adam(lr=0.0001),loss='mse')
        #model.fit(X_train, y_train,epochs=nb_epoch, validation_data = (X_test, y_test))
        model.fit(X_train, y_train,epochs=nb_epoch, batch_size=batch_size)

        model.save(path)

        rospy.loginfo('model saved to: %s', path)

        print('the limits used were: ' + str(limits))


if __name__ == '__main__':
    main()
