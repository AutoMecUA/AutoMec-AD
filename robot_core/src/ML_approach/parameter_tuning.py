
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#   Importing the dataset
dataset1 = pd.read_csv('good_train_data/27_02_21__13_48_11.csv') #200 linhas Driver: Daniel Coelho
#dataset2 = pd.read_csv('good_train_data/27_02_21__13_58_02.csv') #600 linhas Driver: Daniel Coelho
#dataset3 = pd.read_csv('good_train_data/28_02_21__12_18_50.csv') #600 linhas Driver: Tiago Reis
#dataset4 = pd.read_csv('good_train_data/28_02_21__14_22_55.csv') #800 linhas Driver: Tiago Reis


dataset1 = dataset1.drop(columns=['linear','angular','pixel.20400'])
#dataset2 = dataset2.drop(columns=['linear','angular','pixel.20400'])
#dataset3 = dataset3.drop(columns=['linear','angular','pixel.20400'])
#dataset4 = dataset4.drop(columns=['linear','angular','pixel.20400'])
#dataset = dataset1.append(dataset2)
#dataset_train = dataset2.append(dataset3)
#dataset_train = dataset3.append(dataset4)
dataset_train=dataset1



X_train = dataset_train.iloc[:,:-1].values
y_train = dataset_train.iloc[:,-1].values

#X_test = dataset1.iloc[:,:-1].values
#y_test = dataset1.iloc[:,-1].values

# Training the model
from catboost import CatBoostRegressor
regressor = CatBoostRegressor()
regressor.fit(X_train, y_train) 

from sklearn.model_selection import GridSearchCV
parameters = [{'iterations':[1000,2000,500], 'learning_rate' : [0.03,0.05,0.07,0.01,0.005],'depth':[6,10,16,25,3]},
              ]

grid_search = GridSearchCV(estimator= regressor,param_grid=parameters,scoring='r2',n_jobs=-1)

grid_search.fit(X_train,y_train) # now we are doing  multiple trainings
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)