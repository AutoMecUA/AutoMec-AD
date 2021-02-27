#!/usr/bin/env python

# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#   Importing the dataset
dataset1 = pd.read_csv('../good_train_data/27_02_21__13_48_11.csv')
dataset2 = pd.read_csv('../good_train_data/27_02_21__13_58_02.csv')

dataset1 = dataset1.drop(columns=['linear','angular','pixel.20400'])
dataset2 = dataset2.drop(columns=['linear','angular','pixel.20400'])
dataset = dataset1.append(dataset2)
print(dataset.shape)
print(dataset1.shape)
print(dataset2.shape)

X_train = dataset2.iloc[:,:-1]
y_train = dataset2.iloc[:,-1]

X_test = dataset1.iloc[:,:-1]
y_test = dataset1.iloc[:,-1]


# Training the model
from catboost import CatBoostRegressor
regressor = CatBoostRegressor()
regressor.fit(X_train, y_train) 

#Predicting                             
y_pred=regressor.predict(X_test)

#   Evaluating the model performance
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))