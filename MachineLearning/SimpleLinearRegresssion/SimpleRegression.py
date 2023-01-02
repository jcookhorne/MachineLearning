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

# Training the Simple Linear Regression model on the Training Set
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train, y_train)

# Predicting the test set results
y_pred = regression.predict(x_test)



# Visualizing the Training set results
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regression.predict(x_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the Test set Results

# the only that will really change is the plt.scatter because those are the points
# that are on the graph the line should remain the same between both graphs
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regression.predict(x_train), color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
