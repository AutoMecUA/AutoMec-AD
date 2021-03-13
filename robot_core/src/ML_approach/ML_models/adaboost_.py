#!/usr/bin/env python

# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor


#   Importing the dataset
dataset1 = pd.read_csv('../good_train_data_turtle/07_03_21__18_56_30.csv')


dataset1 = dataset1.drop(columns=['linear','angular','pixel.20400'])

dataset = dataset1


X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

#  Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 0) 


# Training the model
from catboost import CatBoostRegressor
regressor = AdaBoostRegressor(random_state=0, n_estimators=100)
regressor.fit(X_train, y_train) 

#Predicting                             
y_pred=regressor.predict(X_test)

#   Evaluating the model performance
from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))