import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2
import random


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.engine.sequential import relax_input_shape


def getName(filePath):
    # Removes the /USers/C/ ... from the Path so we get only the image name
    return filePath.split('\\')[-1]


def importDataInfo(path):
    columns = ['Center',
               'Steering']
    data = pd.read_csv(os.path.join(path, 'driving_log.csv'), names=columns)
    # REMOVE FILE PATH AND GET ONLY FILE NAME
    # print(getName(data['center'][0]))
    data['Center'] = data['Center'].apply(getName)
    # print(data.head())
    print('Total Images Imported', data.shape[0])
    return data


def balanceData(data, display=True):
    # To sum up, since the car runs with 0 Angle most of the time we chunk those data sets off

    nBins = 31  # Has to be an odd number so we have zero at the center
    samplesPerBin = 500  # Change later with more values, ex 1000
    hist, bins = np.histogram(data['Steering'], nBins)
    plt.title("Steering Angle Distribuiton")
    # print(bins)
    # center-  This creates a Bin with zero value for the center (wich is what we expect from most of the driving input)
    if display:
        center = (bins[:-1] + bins[1:])*0.5
        # print(center)
        plt.bar(center, hist, width=0.06)
        plt.plot((-1, 1), (samplesPerBin, samplesPerBin))
        plt.show()  # Nota, eu conduzi mais voltas para o lado esquerdo então nao está balanced

    # To normalize the data since we have to many zeros , we put a threshold (samplesPerBin)
    # And Clear the Data
    removeIndexList = []
    for j in range(nBins):
        # Iterates trough the bins , checks if steering angle fits in the bin, if it's over a certian value it gets deleted
        binDataList = []
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j + 1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeIndexList.extend(binDataList)
    print('Removed Images: ', len(removeIndexList))
    data.drop(data.index[removeIndexList], inplace=True)
    print('Remaining Images: ', len(data))

    if display:
        hist, _ = np.histogram(data['Steering'], nBins)
        plt.bar(center, hist, width=0.06)
        plt.plot((-1, 1), (samplesPerBin, samplesPerBin))
        plt.title("Steering Angle Distribuiton (Trimmed)")
        plt.show()  # Ver Graficos

    return data


def loadData(path, data):

    imagesPath = []
    steering = []

    for i in range(1,len(data)):

        indexedData = data.iloc[i]  # Grabbing 1 entry
        # print(indexedData)
        imagesPath.append(os.path.join(path, 'IMG', indexedData[0]))
        steering.append(float(indexedData[1]))
    imagesPath = np.asarray(imagesPath)
    steering = np.asarray(steering)

    return imagesPath, steering


def augmentImage(imgPath, steering):
    # Function: Add randomness to the data set by applying random "filters"

    img = mpimg.imread(imgPath)

    # PAN
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={
                         'x': (-0.1, 0.1), 'y': (-0.1, 0.1)})
        img = pan.augment_image(img)  # Add a pan to img

    # Zoom
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)

    # Brightness
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.2, 1.2))
        img = brightness.augment_image(img)

    # Flip
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = - steering

    return img, steering


def preProcessing(img):
    # Cropping Region of intrest, Ajust with Gazebo and use Andre Code in the Future
    # img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # For better jornalization
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))  # That's what NIVIDA uses
    img = img/255

    return img


#imgRe = preProcessing(mpimg.imread('testimg.jpg'))
# plt.imshow(imgRe)
# plt.show()


def batchGen(imagesPath, steeringList, batchSize, trainFlag):
    # Creates a batch and applies augmentation
    while True:
        imgBatch = []
        steeringBatch = []

        for i in range(batchSize):
            # Gets a random image and augments it
            index = random.randint(0, len(imagesPath) - 1)
            if trainFlag:
                img, steering = augmentImage(
                    imagesPath[index], steeringList[index])
            else:
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index]
            img = preProcessing(img)

            imgBatch.append(img)
            steeringBatch.append(steering)

        yield(np.asarray(imgBatch), np.asarray(steeringBatch))


def createModel():
    model = Sequential()
    model.add(Convolution2D(24, (5, 5), (2, 2),
                            input_shape=(66, 200, 3), activation='elu'))
    model.add(Convolution2D(36, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), (2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))

    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.compile(Adam(lr=0.0001), loss='mse')
    return model

# PARTE ESPECIFICA DO SIMULADOR UDACITY
