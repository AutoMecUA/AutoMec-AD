#!/usr/bin/env python3

from utils import *
from sklearn.model_selection import train_test_split
import os
import pathlib
import rospy
from tensorflow.keras.models import load_model

import yaml
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():

    # Init Node
    rospy.init_node('ml_training', anonymous=False)

    base_folder = rospy.get_param('~base_folder', 'set1') 
    modelname = rospy.get_param('~modelname', 'model1.h5')
    epochs = rospy.get_param('~epochs', 20)
    steps_per_epoch = rospy.get_param('~steps_per_epoch', 100)
    batch_xtrain = rospy.get_param('~batch_xtrain', 20)
    batch_ytrain = rospy.get_param('~batch_ytrain', 1)
    batch_xval = rospy.get_param('~batch_xval', 25)
    batch_yval = rospy.get_param('~batch_yval', 0)
    validation_steps = rospy.get_param('~validation_steps', 50)

    image_width = rospy.get_param('~width', 320)
    image_height = rospy.get_param('~height', 160)
    # params only used in yaml file
    #env = rospy.get_param('~env', 'gazebo')

    print('base_folder: ', base_folder)
    print('modelname: ', modelname)
    print('epochs: ', epochs)
    print('steps_per_epoch: ', steps_per_epoch)
    print('batch_xtrain: ', batch_xtrain)
    print('batch_ytrain: ', batch_ytrain)
    print('batch_xval: ', batch_xval)
    print('batch_yval: ', batch_yval)
    print('validation_steps: ', validation_steps)

    # TODO - Adicionar conditionals para treinar o modelo se ele já existe

    # STEP 1 - Initialize Data
    s = str(pathlib.Path(__file__).parent.absolute())
    path_data = s + '/../../data/' + base_folder
    data = importDataInfo(path_data + '/')

    #print('\ndata load from ' + base_folder)
    rospy.loginfo('Train with data load from:\n %s', base_folder)

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


    #sys.exit()

    # Step 2 - Vizualize and Balance data
    balanceData(data, display=True)

    # Step 3 - Prepare for Processing
    imagesPath, steerings = loadData(path_data, data)
    #print(imagesPath[0], steerings[0])

    # Step 4 - Split for Training and Validation, 70/30 by default
    xTrain, xVal, yTrain, yVal = train_test_split(
        imagesPath, steerings, test_size=0.3, random_state=5)
    print('Total Training Images: ', len(xTrain))
    print('Total Validation Images: ', len(xVal))


    # Step 5- Augmentation and Variation to the Data
    # Defined Helper function

    # Step 6 - Preprocessing
    # Step 7 - Batch Generator
    # Step 8 - Creating the Model

    path = s + '/../../models/cnn1_' + modelname

    
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

    # Step 9 -Training
    history = model.fit(batchGen(xTrain, yTrain, batch_xtrain, batch_ytrain, image_width, image_height), steps_per_epoch=steps_per_epoch, epochs=epochs,
                        validation_data=batchGen(xVal, yVal, batch_xval, batch_yval, image_width, image_height), validation_steps=validation_steps)

    # Step 10 - Saving and plotting
    #model.save('models_files/' + modelname)
    model.save(path)

    print('\n Model Saved to ' + path)
    print('\n Model Saved to ' + modelname)
    rospy.loginfo('Model Saved to: %s', path)

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

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Training', 'Validation'])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.show()

    # TODO EDITAR ISTO PARA PODER TREINAR MULTIPLAS VEZES

if __name__ == '__main__':
    main()
