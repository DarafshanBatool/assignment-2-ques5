# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 00:13:05 2020

@author: Laiba Masood
"""

# Simple Linear Regression
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\Laiba Masood\Desktop\NCAI ML Course\assignment 2\assignment 2\monthlyexp vs incom.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/2, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Monthly Experience VS Income (Training set)')
plt.xlabel('Months of Experience')
plt.ylabel('Income distribution of employees')
plt.show()
# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'green')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Monthly Experience VS Income (Test set)')
plt.xlabel('Months of Experience')
plt.ylabel('Income distribution of employees')
plt.show()
