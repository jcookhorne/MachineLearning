import numpy as np
import pandas as pd
import tensorflow as tf
# print(tf.__version__)
dataset = pd.read_csv('Churn_Modelling.csv')
#Please remember that Y is dependent
#also know that the first ':' is for rows, and the second is for columns
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

from sklearn.preprocessing import  LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers={('encoder', OneHotEncoder(), [1])}, remainder='passthrough')
X = np.array(ct.fit_transform(X))
