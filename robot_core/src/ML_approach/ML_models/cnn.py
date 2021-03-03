
from sklearn.datasets import load_boston
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset2 = pd.read_csv('../good_train_data/27_02_21__13_58_02.csv') #600 linhas Driver: Daniel Coelho
dataset3 = pd.read_csv('../good_train_data/28_02_21__12_18_50.csv') #600 linhas Driver: Tiago Reis
dataset4 = pd.read_csv('../good_train_data/28_02_21__14_22_55.csv') #800 linhas Driver: Tiago Reis

dataset2 = dataset2.drop(columns=['linear','angular','pixel.20400'])
dataset3 = dataset3.drop(columns=['linear','angular','pixel.20400'])
dataset4 = dataset4.drop(columns=['linear','angular','pixel.20400'])

dataset = dataset2.append(dataset3)
dataset = dataset.append(dataset4)

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

X = X.reshape(X.shape[0], X.shape[1], 1)

#  Splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 0) 

model = Sequential()
model.add(Conv1D(32, 2, activation="relu", input_shape=(X.shape[1],1)))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(1))
model.compile(loss="mse", optimizer="adam")

model.fit(X_train, y_train, batch_size=12,epochs=100, verbose=0)

y_pred = model.predict(X_test)

print("MSE: %.4f" % mean_squared_error(y_test, y_pred))