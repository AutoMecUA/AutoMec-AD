#!/usr/bin/env python3

import yaml
import os
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

def createModel(imgwidth, imgheight):
    model = Sequential()
    model.add(Convolution2D(24, (5,5),(2, 2), input_shape=(imgheight, imgwidth, 1), activation='elu'))  
    model.add(Convolution2D(36,(5,5),(2, 2), activation='elu'))
    model.add(Convolution2D(48, (5,5),(2, 2), activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Convolution2D(64, 3, 3, activation='elu'))

    model.add(Dropout(0.8))
    model.add(Flatten())
    model.add(Dense(100,activation='elu'))
    model.add(Dense(50,activation='elu'))
    model.add(Dense(10,activation='elu'))
    model.add(Dense(2))
    return model


def main():

    # Init Node
    rospy.init_node('ml_training', anonymous=False)

    base_folder = rospy.get_param('~base_folder', 'set1') 
    modelname = rospy.get_param('~modelname', 'model1.h5')
    nb_epoch = rospy.get_param('~epochs', 20)
    batch_size = rospy.get_param('~batch_size', 32)

    image_width = rospy.get_param('~width', 320)
    image_height = rospy.get_param('~height', 160)

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

    # yaml
    if not os.path.isfile(path_data + '/info.yaml'):
        have_dataset_yaml = False
        # we may allow to continue processing with default data
        print("no yaml info file found. exit.")
        sys.exit()
    else:
        have_dataset_yaml = True

    # dataset yaml defaults
    ds_cam_angle = ''
    ds_cam_height = ''
    ds_developer = ''
    ds_environment = ''
    ds_frequency = 0
    ds_image_size = ''
    ds_linear_velocity = 0

    if have_dataset_yaml:
        with open(path_data + '/info.yaml') as file:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            info_loaded = yaml.load(file, Loader=yaml.FullLoader)

            ds_cam_angle = info_loaded['dataset']['cam_angle']
            ds_cam_height = info_loaded['dataset']['cam_height']
            ds_developer = info_loaded['dataset']['developer']
            ds_environment = info_loaded['dataset']['environment']
            ds_frequency = info_loaded['dataset']['frequency']
            ds_image_size = info_loaded['dataset']['image_size']
            ds_linear_velocity = info_loaded['dataset']['linear_velocity']

    # Writing CSV
    total_left_angles = 0
    total_right_angles = 0
    total_straight_angles = 0

    images = []
    angles = []
    velocity = []
    with open(path_data + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        k = 0
        for line in reader:
            # k += 1
            # if (k%2)==0:
            #     continue
            if k==1: continue

            center_image = cv2.imread(image_path + line[0],0)
            center_image = cv2.resize(center_image, (320, 160))
            center_image_temp = np.expand_dims(center_image, axis=2)

            images.append(center_image_temp)
            angles.append(float(line[1]))
            velocity.append(float(line[2]))
            
            # Change image and store the same angle, with some distortion, both in image and in angle (to increase robustness)
            # flipped
            center_image_temp = np.expand_dims(cv2.flip(center_image, 1), axis=2)

            images.append(center_image_temp)
            angles.append(-float(line[1]))
            velocity.append(float(line[2]))
            # no more distortions
            
            if(float(line[1]) < -0.1):
                total_left_angles += 1
            elif(float(line[1]) > 0.1):
                total_right_angles += 1
            else:
                total_straight_angles += 1


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

    images = np.asarray(images)
    images = images.astype('float32')
    images /= 255
    target = np.column_stack((angles,velocity))

    print('####################################3')
    print(target[:,1])
    print('####################################3')


    X_train, X_test, y_train, y_test = train_test_split(images, target, test_size=0.00001)

    train_samples_size = len(X_train)
    validation_samples_size = len(X_test)

    # print('####################################3')
    # print(y_train[0])
    # print('####################################3')

    total_left_angles = 0
    total_right_angles = 0
    total_straight_angles = 0

    for train_sample in y_train[:,0]:
        if(float(train_sample) < -0.15):
            total_left_angles += 1
        elif(float(train_sample) > 0.15):
            total_right_angles += 1
        else:
            total_straight_angles += 1

    left_to_straight_ratio = 0
    right_to_straight_ratio = 0

    left_to_straight_ratio = total_straight_angles/total_left_angles
    right_to_straight_ratio = total_straight_angles/total_right_angles

    print()
    print('Train Sample Size : ', train_samples_size)
    print('Validation Sample Size : ', validation_samples_size)

    print()
    print('After Train-Test split, Angle Distribution')
    print('Total Left Angles : ', total_left_angles)
    print('Total Right Angles : ', total_right_angles)
    print('Total Straight Angles : ', total_straight_angles)
    print('Left to Straight Ratio : ', left_to_straight_ratio)
    print('Right to Straight Ratio : ', right_to_straight_ratio)

    path = path_model + 'cnn2av_' + modelname

    enter_pressed = input("\n" + "Create a new model from scratch? [Y/N]: ")

    if enter_pressed.lower() == "y" or enter_pressed == "":
        model = createModel(image_width, image_height)
        is_newmodel = True
    else:
        print('\n Model load from ' + modelname)
        model = load_model(path)
        is_newmodel = False

    # yaml
    if is_newmodel:
        imgsize_list = [image_width, image_height]
        string_ints = [str(int) for int in imgsize_list]
        imgsize_str = ",".join(string_ints)
        # dataset yaml defaults
        model_developer = os.getenv('automec_developer')
        model_image_size = imgsize_str
        model_frequency = ds_frequency
        model_linear_velocity = ds_linear_velocity
        model_environment = ds_environment
        model_cam_angle = ds_cam_angle
        model_cam_height = ds_cam_height
        model_cnn_number = '1'
    else:
        with open(path + '_info.yaml') as file:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            info_loaded = yaml.load(file, Loader=yaml.FullLoader)

            model_developer = info_loaded['model']['developer']
            model_image_size = info_loaded['model']['image_size']
            model_frequency = info_loaded['model']['frequency']
            model_linear_velocity = info_loaded['model']['linear_velocity']
            model_environment = info_loaded['model']['environment']
            model_cam_angle = info_loaded['model']['cam_angle']
            model_cam_height = info_loaded['model']['cam_height']
            model_cnn_number = info_loaded['model']['cnn_number']

    model.summary()

    model.compile(Adam(lr=0.0001),loss='mse')
    #model.fit(X_train, y_train,epochs=nb_epoch, validation_data = (X_test, y_test))
    model.fit(X_train, y_train,epochs=nb_epoch, batch_size=batch_size)

    model.save(path)

    rospy.loginfo('model saved to: %s', path)

    # yaml
    info_data = dict(

        model = dict(
            developer = os.getenv('automec_developer'),
            image_size = model_image_size,
            frequency = model_frequency,
            linear_velocity = model_linear_velocity,
            environment = model_environment,   
            cam_height = model_cam_height,
            cam_angle = model_cam_angle,
            cnn_number = model_cnn_number
        )
    )

    with open(path + '_info.yaml', 'w') as outfile:
        yaml.dump(info_data, outfile, default_flow_style=False)

    rospy.loginfo('yaml Saved')


if __name__ == '__main__':
    main()
