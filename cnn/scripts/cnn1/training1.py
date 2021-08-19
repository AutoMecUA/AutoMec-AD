#!/usr/bin/env python3

from utils import *
from sklearn.model_selection import train_test_split
import os
import pathlib
import rospy
from tensorflow.keras.models import load_model

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

    print("\n" + "Create a new model from scratch? [Y/N]")
    if input().lower() == "y":
        model = createModel()
    else:
        print('\n Model load from ' + modelname)
        model = load_model(path)

    model.summary()

    # Step 9 -Training
    history = model.fit(batchGen(xTrain, yTrain, batch_xtrain, batch_ytrain), steps_per_epoch=steps_per_epoch, epochs=epochs,
                        validation_data=batchGen(xVal, yVal, batch_xval, batch_yval), validation_steps=validation_steps)

    # Step 10 - Saving and plotting
    #model.save('models_files/' + modelname)
    model.save(path)
    print('\n Model Saved to ' + path)
    print('\n Model Saved to ' + modelname)
    rospy.loginfo('Model Saved to: %s', path)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Training', 'Validation'])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.show()

    # TODO EDITAR ISTO PARA PODER TREINAR MULTIPLAS VEZES

if __name__ == '__main__':
    main()
