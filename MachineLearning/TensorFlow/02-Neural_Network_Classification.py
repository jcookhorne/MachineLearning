"""  Notes
Binary classification
is something 1 thing or the other, (black or white)

multiclass Classification - question like is this a phot of sushi steak or pizza
has more than one item going on but still yes or no for each item

multilabel classification - multiple different options for each item
what were going to cover
architecture of a neural network classification model
input shapes and output shapes of a classification model
creating custom data to view and fit
steps in modelling
creating a model, compiling a model, fitting a model, evaluating a model
Different classification evaluation methods
saving and loading models
EXPERIMENT EXPERIMENT EXPERIMENT

Batch sizes are usually defaulted to 32  meaning it will look
at 32 items in a batch and go through them if their are more you
can set batch size in your tensor

A regression problem focuses on predicting numerical outcomes, but within classification
we are assigning an object to a predefined classes
"""

# **Introduction to neural network classification with Tensorflow
#  in this notebook were going to learn how to write neural
#  networks for classification problems

#  ** a Classification is where you try to classify something as one thing or another
# * a few types of classification problems
# * Binary Classification
# * Multiclass Classification
# * Multilable classification


# * Creating data to view and fit

from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
# make 1000 examples
n_samples = 1000

# create circles
X, y = make_circles(n_samples, noise=0.03, random_state=42)

# Check out the features
print(X)

# Check out the labels
print(y[:10])

# * our data is a little hard to understand right now.. lets visulize
circles = pd.DataFrame({"X0": X[:, 0], "X1": X[:, 1], "label": y})
print(circles)

# visualize with a plot
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()


# input and output shapes of our features and labels
print(X.shape, y.shape)

# how many samples were working with
print(len(X), len(y))

# View the first example of features and labels
print(X[5], y[0])

#set the random seed
tf.random.set_seed(42)

# create the model using the sequential api
model = tf.keras.Sequential([
    # tf.keras.Input(shape=(1000, 1000, 2)),
    # tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(1)
])

# compile the model
model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["accuracy"])

model.fit(X, y, epochs=200, verbose=0)
model.evaluate(X,y)

# we are working on a binary classification problem and our model
# is getting around 50% accuracy its performing as if it is guessing
# so lets add another layer


# set random seed
model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(1)
])

# 2 compile the model
model_2.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.SGD(),
                metrics=["accuracy"])
# 3.fit the model
model_2.fit(X, y, epochs=100, verbose=0)

# 4 evaluate the model
model_2.evaluate(X, y)

# didn't get any better with just adding another layer
# Improving our model!

# 1. Create a model - we might to add more layers or increase the number of hidden units within a layer.
# 2. Compiling a model - here we might to choose a different optimization function such as Adam instead of SGD
# 3. Fitting a model - perhaps we might fit our model for more epochs(leave it training for longer

# create the model 3layers
model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

# 2 compile the model
model_3.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# fit the model
model_3.fit(X,y, epochs=100, verbose=0)
# evaluate the modle
model_3.evaluate(X, y)
model_3.predict(X)

# Take in a trained model, features X and labels y
# create a meshgrid of different X values
# make predictions across the meshgrid
# plot the predictions as well as a line between zones ( where each unique class falls)


