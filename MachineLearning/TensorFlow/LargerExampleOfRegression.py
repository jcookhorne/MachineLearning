# import required Libraries
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# read in the insurance dataset
insurance = pd.read_csv("insurance.csv")
# print(insurance)
# print(insurance["smoker"])

# dummy indicator variables is just one hot encoding
# so taking non numerical data and turning it numerical
# get dummies returns boolean naturally so cast the type to int
insurance_one_hot = pd.get_dummies(insurance, dtype=int)
# set options is how you keep it from truncating the data you are working with
# print(insurance_one_hot.head(), pd.set_option("display.max_columns", 12))

# Create X & Y values
X = insurance_one_hot.drop("charges", axis=1)
y = insurance_one_hot["charges"]

# print(X.head())
# print(y.head())
# Create training and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(len(X), len(X_train), len(X_test))
# print(len(y), len(y_train), len(y_test))

# build a neural network
tf.random.set_seed(42)
# 1. Create a model
insurance_model = tf.keras.Sequential([
    tf.keras.layers.Dense(17),
    tf.keras.layers.Dense(1)
])
# 2. Compile the model
insurance_model.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.SGD(),
                        metrics=["mae"])
# 3. fit the model
# insurance_model.fit(X_train,  y_train, epochs=100)

# Check the results of the insurance model on the test data
print("evaluating the test between x and y: ", insurance_model.evaluate(X_test, y_test))
print("Finding the median of the y train data: ", y_train.median(), "not the mean: ", y_train.mean())

# trying to improve our mode
# 1. add an extra layer with more hidden units
# 2. train for longer
# 3. change the optimizer

# 1. create the model
insurance_model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(17),
    tf.keras.layers.Dense(1)
])

# 2. compile the model
insurance_model_2.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(),
                          metrics=["mae"])

# 3. fit the model
insurance_model_2.fit(X_train, y_train, epochs=100, verbose=0)

# evaluate the larger model
print("Evaluating 2nd model: ", insurance_model_2.evaluate(X_test, y_test))

# 1. create the model
insurance_model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(17),
    tf.keras.layers.Dense(1)
])

# 2. compile the model
insurance_model_3.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(),
                          metrics=["mae"])

# 3. fit the model
# history = insurance_model_3.fit(X_train, y_train, epochs=400)

print("Evaluating 3rd model", insurance_model_3.evaluate(X_test, y_test))

# Plot history ( also known as a loss curve or a training curve)
# pd.DataFrame(history.history).plot()
# plt.ylabel("loss")
# plt.xlabel("epochs")
# plt.show()

# ** how long should you train for (it depends really . . . it depends
# on the problem you're working on. However, many people have asked this question ebfore
# ... so Tensorflow has a solution! It;s called the EarlyStopping Callback)

# ** Preprocessing data (Normalization and standardization)

# Steps in modelling with tensorflow
# turn all data into numbers one hot encoding
# make sure all of your tensors are the right shape
# Scale features ( normalize of standardize, neural networks tend to prefer normalization)

# X["age"].plot(kind="hist")
# X["children"].plot(kind="hist")
# plt.show()

# scale or normalization converts all values to between 0 and 1 while preserving the original distribution

# to prepare our data, we can borrow a few classes for scikit-learn

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
# create a column transformer
ct = make_column_transformer((MinMaxScaler(), ["age", "bmi", "children"]),  # turn all values in
                             # these columns between 0 and 1
                             (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])
                             )
# Create X and y
X2 = insurance.drop("charges", axis=1)
y2 = insurance["charges"]

# Build our traina nd test sets

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Fit the column transformer to our training data
ct.fit(X2_train)
# Transform training and test data with normalization (MinMaxScalar) and OneHotEncoder
X2_train_normal = ct.transform(X2_train)
X2_test_normal = ct.transform(X2_test)

# what does our data look like now?
# print(X2_train.loc[0])
# print(X2_train_normal[0])
# print(X2_train.shape, X2_train_normal.shape)

# Build a neural network model to fit on our normalized data
tf.random.set_seed(42)

# 1 create the model
# insurance_model_2.summary()
insurance_model_4 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

# 2 compile the model
insurance_model_4.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(),
                          metrics=["mae"])

# 3. Fit the model
insurance_model_4.fit(X2_train_normal, y2_train, epochs=100, verbose=0)

# Evaluate our insurance model trained on normalized data
print("Evaluating model 4", insurance_model_4.evaluate(X2_test_normal, y2_test))




