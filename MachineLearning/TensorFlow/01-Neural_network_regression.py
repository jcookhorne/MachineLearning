import matplotlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# * Introduction to Regression with neural Networks in Tensorflow

# * there are many defintions for a regression problem but in our case, were going to simplify it:
# * predicting a numerical variable based on some other combination of variables, even shorter... predicting a number

# * check tensorflow version
# print(tf.__version__)

# * Create features
X1 = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0,  11.0, 14.0])
# * Create labels ( usually y in lowercase)
y1 = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

# Visualize it

# plt.scatter(X, y)
# plt.show(block=True)

# print(y == X + 10)
# * input and output shapes
# * Create a demo tensor for out housing price prediction problem

house_info = tf.constant(["bedroom", "bathroom", "garage"])
house_price = tf.constant([939700])
# print(house_info, house_price)

input_shape = X1.shape
output_shape = y1.shape
# print(input_shape, output_shape)
# print(X[0], y[0])
# print(X[1], y[1])
input_shape1 = X1[0].shape
output_shape1 = y1[0].shape
# print(input_shape, output_shape)
# print(X[0].ndim)


# * turn our numpy array into tensor

X = tf.cast(tf.constant(X1), dtype=tf.float32)
y = tf.cast(tf.constant(y1), dtype=tf.float32)
# print(X, y)

# print(input_shape1, output_shape1)

# * Steps in modelling with Tensorflow
# * 1) Creating a model = define the input and output layers, as well as the hidden layers of a deep learning model.
# * 2) Compiling a model = define  the loss function  ( in other words, the function which tells our model how wrong it
# * is and the optimizer ( tells our model how to improve the patterns its learning) and evaluation metrics (what we can
# * use to interpret the performance of our model).
# * 3) Fitting a model - letting the model try to find patterns between X & y (features and labels).

# * set a random seed
tf.random.set_seed(41)

# * 1) create a model using the Sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

# * another way to use sequential api each add would be the same as adding it to the array above!
# model1 = tf.keras.Sequential()
# model1.add(tf.keras.layers.Dense(1))

# * 2) compile the modle
# * mae is short for mean absolute error
# * SGD is short for stochastic gradient descent
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

# * 3) fit the model
# print(X, y)
# model.fit(X, y, epochs=5)
# you cannot have a model with values that have no dimensions aka you need at least 2 and that goes for both
model.fit(tf.expand_dims(X1, axis=-1), tf.expand_dims(y1, axis=-1), epochs=5)

# Check out X and Y
print(X1, y1)
# * try and make a prediction
print(model.predict(x=tf.cast([17.0], dtype=tf.float32)))


# * Improve our model
