import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential

print(tf.__version__)
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
# print(X)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

#this is splitting into a trinaing and a test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#this is for feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(X_train)
x_test = sc.transform(X_test)

#this is usings tensorflow to initialize an Object that will be our artificial neural network
ann = Sequential()
#now using the ann we will make our first input layer
#can add a hidden or a dropout layer not just a fully connected layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
#adding a second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
#adding the output layer: for that you need the sigmoid activation function
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
#if there was more than 2 fields it wouldnt be sigmoid but rather softmax

#part 3 training the ANN

#loss is binary only because it is for binary data aka 2 fields)
#if we had more than 2 data fields it would be crossentropyless
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#training the ANN on the training set
ann.fit(x_train, y_train, batch_size=32, epochs=20)

#any input of the predict method must be in a double square bracket
#so a 2D Array
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

#predicting test set results
y_pred = ann.predict(x_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))


#making the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))


