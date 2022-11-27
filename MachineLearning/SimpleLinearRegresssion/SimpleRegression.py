import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing in the dataset
from sklearn.impute import SimpleImputer
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(x)
print(y)
# Splitting the dataset into the Training set
# and the Test set
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#Training the Simple Linear Regression model on the Training Set
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)


