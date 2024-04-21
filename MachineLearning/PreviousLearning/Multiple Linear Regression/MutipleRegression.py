import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print(x)
print(y)

#Encoding the Categorical Data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

print(x)
#Multiple regression class will build and train whihc
# will avoid the dummy model trap

#Splitting Training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0
                                                    )
#Training train_set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
#LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,normalize=False )

# print("X training Set: ", x_train)
# print("X Test set: ", x_test)
#
# print("Y Training set: ", y_train)
# print("Y Test set: ", y_test)
#predicting The test set


y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
#to display the vector data vertically use .reshape
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)),1))

