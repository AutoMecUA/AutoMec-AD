#!/usr/bin/env python

# Random Forest Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#   Importing the dataset
#dataset1 = pd.read_csv('../good_train_data_turtle/07_03_21__18_56_30.csv')
dataset2 = pd.read_csv('../good_train_data_turtle/13_03_21__16_17_03.csv')
#dataset3 = pd.read_csv('../good_train_data_turtle/11_03_21__20_17_43.csv')
#dataset4 = pd.read_csv('../good_train_data_turtle/11_03_21__20_19_04.csv')

#dataset1 = dataset1.drop(columns=['linear','angular','pixel.20400'])
dataset2 = dataset2.drop(columns=['linear','angular','pixel.20400'])
#dataset3 = dataset3.drop(columns=['linear','angular','pixel.20400'])
#dataset4 = dataset4.drop(columns=['linear','angular','pixel.20400'])

print('excel done')


#dataset = dataset2.append([dataset3,dataset4])
dataset=dataset2
print('append off')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

print('train begin')

# Training the model
from catboost import CatBoostRegressor
regressor = CatBoostRegressor()
regressor.fit(X, y) 

regressor.save_model('catboost_file_turtle_test2')                       




