#!/usr/bin/env python

# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#   Importing the dataset
dataset1 = pd.read_csv('/home/trsr/28_02_21__21_11_06.csv') #2000 linhas Driver: Tiago Reis
dataset2 = pd.read_csv('/home/trsr/28_02_21__21_15_15.csv') #2000 linhas Driver: Tiago Reis
dataset3 = pd.read_csv('/home/trsr/28_02_21__21_18_21.csv') #2000 linhas Driver: Tiago Reis
dataset4 = pd.read_csv('/home/trsr/28_02_21__21_21_24.csv') #2000 linhas Driver: Tiago Reis
dataset5 = pd.read_csv('/home/trsr/28_02_21__21_24_59.csv') #2000 linhas Driver: Tiago Reis

print("Excel accepted")

dataset1 = dataset1.drop(columns=['linear','angular','pixel.20400'])
dataset2 = dataset2.drop(columns=['linear','angular','pixel.20400'])
dataset3 = dataset3.drop(columns=['linear','angular','pixel.20400'])
dataset4 = dataset4.drop(columns=['linear','angular','pixel.20400'])
dataset5 = dataset5.drop(columns=['linear','angular','pixel.20400'])

print("Data drop")

dataset = dataset1.append([dataset2,dataset3,dataset4,dataset5])

print("Big Data")

#print(dataset.shape)


X_train = dataset.iloc[:,:-1].values
y_train = dataset.iloc[:,-1].values

# X_test = dataset1.iloc[:,:-1].values
# y_test = dataset1.iloc[:,-1].values
X_test = dataset5.iloc[:,:-1].values
y_test = dataset5.iloc[:,-1].values



# Training the model
from catboost import CatBoostRegressor
regressor = CatBoostRegressor()
regressor.fit(X_train, y_train) 

#Predicting                             
y_pred=regressor.predict(X_test)

#   Evaluating the model performance
from sklearn.metrics import r2_score
# print(r2_score(y_test,y_pred))
regressor.save_model('catboost_file_test')
# print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))