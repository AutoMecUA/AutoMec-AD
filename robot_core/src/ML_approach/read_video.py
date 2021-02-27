#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2 

from csv import writer
from datetime import datetime

import csv
dataset1 = pd.read_csv('good_train_data/27_02_21__13_48_11.csv')
dataset1 = dataset1.drop(columns=['linear','angular','pixel.20400','pixel.20401'])

for i in range(0,dataset1.shape[1]):

    new_data = dataset1.iloc[i,:]
    dict_map={0:0,1:255}
    new_data = new_data.map(dict_map)
    new_data = new_data.values
    final_array=np.resize(new_data,(120,170))
    #resized = cv2.resize(final_array.astype(np.uint8), (680,480), interpolation = cv2.INTER_AREA)
    cv2.imshow('video',final_array.astype(np.uint8))
    cv2.waitKey(100)

