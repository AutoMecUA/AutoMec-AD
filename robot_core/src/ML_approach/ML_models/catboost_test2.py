#!/usr/bin/env python

# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#   Importing the dataset
dataset1 = pd.read_csv('../good_train_data/06_03_21__15_22_21.csv') #2000 linhas Driver: Tiago Reis
# dataset2 = pd.read_csv('../good_train_data/27_02_21__13_48_11.csv') #2000 linhas Driver: Tiago Reis
# dataset3 = pd.read_csv('../good_train_data/27_02_21__13_58_02.csv') #2000 linhas Driver: Tiago Reis
# dataset4 = pd.read_csv('../good_train_data/28_02_21__12_18_50.csv') #2000 linhas Driver: Tiago Reis
# dataset5 = pd.read_csv('../good_train_data/28_02_21__14_22_55.csv') #2000 linhas Driver: Tiago Reis
# dataset6 = pd.read_csv('../good_train_data/28_02_21__20_26_31.csv')
# dataset7 = pd.read_csv('../good_train_data/28_02_21__20_32_02.csv')

print("Excel accepted")

dataset1 = dataset1.drop(columns=['linear','angular','pixel.20400'])
# dataset2 = dataset2.drop(columns=['linear','angular','pixel.20400'])
# dataset3 = dataset3.drop(columns=['linear','angular','pixel.20400'])
# dataset4 = dataset4.drop(columns=['linear','angular','pixel.20400'])
# dataset5 = dataset5.drop(columns=['linear','angular','pixel.20400'])
# dataset6 = dataset6.drop(columns=['linear','angular','pixel.20400'])
# dataset7 = dataset7.drop(columns=['linear','angular','pixel.20400'])

print("Data drop")

# dataset = dataset1.append([dataset2,dataset3,dataset4,dataset5,dataset6,dataset7])

print("Big Data")

#print(dataset.shape)


X_train = dataset1.iloc[:,:-1].values
y_train = dataset1.iloc[:,-1].values

# X_test = dataset1.iloc[:,:-1].values
# y_test = dataset1.iloc[:,-1].values
#X_test = dataset5.iloc[:,:-1].values
#y_test = dataset5.iloc[:,-1].values



# Training the model
from catboost import CatBoostRegressor
regressor = CatBoostRegressor()
regressor.fit(X_train, y_train) 

#Predicting                             
#y_pred=regressor.predict(X_test)

#   Evaluating the model performance
from sklearn.metrics import r2_score
# print(r2_score(y_test,y_pred))
regressor.save_model('catboost_file_new_robot2')
# print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))