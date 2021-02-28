#!/usr/bin/env python

# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#   Importing the dataset
dataset1 = pd.read_csv('../good_train_data/27_02_21__13_48_11.csv') #200 linhas Driver: Daniel Coelho
dataset2 = pd.read_csv('../good_train_data/27_02_21__13_58_02.csv') #600 linhas Driver: Daniel Coelho
dataset3 = pd.read_csv('../good_train_data/28_02_21__12_18_50.csv') #600 linhas Driver: Tiago Reis
dataset4 = pd.read_csv('../good_train_data/28_02_21__14_22_55.csv') #800 linhas Driver: Tiago Reis


dataset1 = dataset1.drop(columns=['linear','angular','pixel.20400'])
dataset2 = dataset2.drop(columns=['linear','angular','pixel.20400'])
dataset3 = dataset3.drop(columns=['linear','angular','pixel.20400'])
dataset4 = dataset4.drop(columns=['linear','angular','pixel.20400'])
dataset = dataset1.append(dataset2)
dataset_train = dataset2.append(dataset3)
dataset_train = dataset_train.append(dataset4)

print(dataset.shape)
print(dataset1.shape)
print(dataset2.shape)
print(dataset_train.shape)

X_train = dataset_train.iloc[:,:-1].values
y_train = dataset_train.iloc[:,-1].values

X_test = dataset1.iloc[:,:-1].values
y_test = dataset1.iloc[:,-1].values



# Training the model
from catboost import CatBoostRegressor
regressor = CatBoostRegressor()
regressor.fit(X_train, y_train) 

#Predicting                             
y_pred=regressor.predict(X_test)

#   Evaluating the model performance
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))
regressor.save_model('catboost_file')
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))