#!/usr/bin/env python

# Random Forest Regression

# Importing the libraries
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#   Importing the dataset

dataset1 = pd.read_csv('../csv_data/20_03_21__16_18_5920_20.csv')
dataset2 = pd.read_csv('../csv_data/20_03_21__20_59_4920_20.csv')
dataset3 = pd.read_csv('../csv_data/20_03_21__21_09_4120_20.csv')
dataset4 = pd.read_csv('../csv_data/20_03_21__21_21_0420_20.csv')


print("Excel accepted")

dataset = dataset1.append(dataset2,ignore_index=True)
dataset = dataset.append(dataset3,ignore_index=True)
dataset = dataset.append(dataset4,ignore_index=True)



X_train = dataset.iloc[:,:-1].values
y_train = dataset.iloc[:,-1].values




# Training the model
from catboost import CatBoostRegressor
regressor = CatBoostRegressor()
regressor.fit(X_train, y_train) 



# Evaluating the model performance
from sklearn.metrics import r2_score
# print(r2_score(y_test,y_pred))
now = datetime.now()
time_now = now.strftime("%H_%M_%S")
model_name = now.strftime("%d") + "_" + now.strftime("%m") + "_" + now.strftime("%y") + "__" + time_now
regressor.save_model('catboost_file_turtle_' + str(model_name) +'_2020')
# print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))