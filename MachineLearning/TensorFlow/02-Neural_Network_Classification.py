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
# * Multilabel classification


# * Creating data to view and fit

from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

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

# set the random seed
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
model.evaluate(X, y)

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
model_3.fit(X, y, epochs=100, verbose=0)
# evaluate the modle
model_3.evaluate(X, y)
model_3.predict(X)


# to visualize our model's predictions, lets create a function plot_decision_boundary(), this function will:
# Take in a trained model, features X and labels y
# create a meshgrid of different X values
# make predictions across the meshgrid
# plot the predictions as well as a line between zones ( where each unique class falls)

def plot_decision_boundary(model, X, y):
    """ plots the decision boundary created by a model predicting on X
    """
    # Define the axis boundaries of the plot and create a meshgrid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    print(x_min, x_max, y_min, y_max)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100));

    # create X values were going to make predictions on these
    x_in = np.c_[xx.ravel(), yy.ravel()] # stack 2d arrays together

    # Make predictions
    y_pred =  model.predict(x_in)

    # check for multi class classification problems
    if len(y_pred[0]) > 1:
        print("doing multiclass classification")
        # we have to reshape our predictions to get them ready for plotting
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print("doing binary classification")
        y_pred = np.round(y_pred).reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    # plt.show()


# Check out the predictions our models are making

# plot_decision_boundary(model=model_3, X=X, y=y)


# lets see if our model can be used for a regression problem...
tf.random.set_seed(42)

# create some regression data
X_regression = tf.range(0, 1000, 5)
y_regression = tf.range(100, 1100, 5) # y =X +100

# print(X_regression, y_regression)

X_reg_train = tf.expand_dims(X_regression[:150], 1)
X_reg_test = tf.expand_dims(X_regression[150:], 1)
y_reg_train = tf.expand_dims(y_regression[:150], 1)
y_reg_test = tf.expand_dims(y_regression[150:], 1)

# Fit our model to the regression data
# model_3.fit(X_reg_train, y_reg_train, epochs=100)

# oh wait we compiles our model for a binary classification problem
# but were now working on a regression problem lets change the model to suit our data


# create the model 3layers
model_4 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

# 2 compile the model, this time with a regression-specific loss function
model_4.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["mae"])

# fit the model
model_4.fit(X_reg_train, y_reg_train, epochs=100)
# make predictions with our trained model
y_reg_preds = model_4.predict(X_reg_test)
plt.figure(figure=(10, 7))
plt.scatter(X_reg_train, y_reg_train, c="b", label="Training data")
plt.scatter(X_reg_test, y_reg_test, c="g", label="Test data")
plt.scatter(X_reg_test, y_reg_preds, c="r", label="Prediction data")
plt.legend()
# plt.show()


# #The missing piece non-linearity
# create the model

model_5 = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation=tf.keras.activations.linear)
])

# 2. Compile
model_5.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=["accuracy"])


# 3 fit the model
history = model_5.fit(X, y, epochs=100)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.show()
plot_decision_boundary(model=model_5, X=X, y=y)


