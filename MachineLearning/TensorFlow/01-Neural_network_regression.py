import keras.src.saving.saving_api
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as img
from keras.src.utils import plot_model

# * Introduction to Regression with neural Networks in Tensorflow

# * there are many defintions for a regression problem but in our case, were going to simplify it:
# * predicting a numerical variable based on some other combination of variables, even shorter... predicting a number

# * check tensorflow version
# print(tf.__version__)

# * Create features
X1 = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0])
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
# * only 5 epochs
# model.fit(tf.expand_dims(X1, axis=-1), tf.expand_dims(y1, axis=-1), epochs=5)

# * epochs being raised to 100
# model.fit(tf.expand_dims(X1, axis=-1), tf.expand_dims(y1, axis=-1), epochs=100)

# Check out X and Y
# print(X1, y1)

# * try and make a prediction
# print(model.predict(x=tf.cast([17.0], dtype=tf.float32)))

# * steps in modelling with tensorFLow
# * 1) Construct or import a pretrained model relevant to your problem
# * 2) Compile the model (prepare it to be used with data)
# ** Loss - how wrong your model's predictions are compared to the truth labels(you want to minimise this).
# ** Optimizer - how your model should update its internal patterns to better its predictions
# ** Metrics - human interpretable values for how well your model is doing
# * 3) Fit the model to the training data so it can discover patterns
# ** Epochs - how many times the model will go through all of the training examples.
# * 4) Evaluate the model on the test data ( how reliable are our model's predictions)

# ** Improving our model
# * we can improve our model, by altering the steps we took to create a model.
# * 1)  Creating a model - here we might add more layers, increase the number of hidden units ( all called neurons)
# * - within each of the hidden layers, change the activation function of each layer
# * 2) Compiling a model - here we might change the optimization function or perhaps the
# learning rate of the optimization function
# * 3) Fitting a  model - here we might fit a model for more epochs ( leave it training longer) or on more data
# (give the model more examples to learn from).

# more changes


# Create a new model with an extra hidden layer with 100 hidden units
newModel = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation=None),
    tf.keras.layers.Dense(1)
])

# 2) compile the model
# newModel.compile(loss="mae",
#                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
#                  metrics=["mae"])

# 3 fit the model
# newModel.fit(tf.expand_dims(X1, axis=-1), tf.expand_dims(y1, axis=-1), epochs=100)

# print(X,'\n', y)
# print(newModel.predict(x=tf.cast([17.0], dtype=tf.float32)))

# although for whatever reason the data we get back is different from the video the knowledge is the same
# a good way to improve a deep model would be to add more hidden layers
# * another way is to increase the number of hidden units
# * change the activation functions
# * change the optimization function
# * change your learning rate - most important hyperparameter of your neural network
# * Fitting on more data
# * fitting for longer

# Evaluating a models performance

# * in practice, a typical workflow you'll go through when building neural network is:
# * build a model -> fit it -> evaluate it -> tweak a model -> fit it -> evaluate it -> tweak a model -> fit it....

# * when it comes to evaluation ... there are 3 words you should memorize
# * visualize visualize visualize
# * its a good idea to visualize
# * the data - what data are we working with what does it look like?
# ** the model itself - what does our model look like?
# * the training of a model - how does a model perform while it learns ?
# * the predictions of the model - how do the predictions of a model line up against the ground truth
# * (the original label)

# ** make a bigger dataset

X2 = tf.range(-100, 100, 4)

y2 = X2 + 10

# print(X2, y2)
# plt.plot(X2, y2)
# plt.scatter(X2, y2)
# plt.show()

# * the 3 sets . . .

# * Training set - the model learns from this data, which is typically 70-80% of the total data you have available.\
# * Validation set - the model gets tuned on this data, which is typically 10-15% of the data available.
#  * Test set - the model gets evaluated on this data to test what is has learned, this set is typically 10-15% of
#  the total data available.
# * you don't always need the validation set


# * check the length of how many samples we have
# print(len(X2))

# * Split the data into train and test sets

X_train = tf.expand_dims(X2[:40], 1)  # the first 40 are training samples ( 8_%  of the data)
y_train = tf.expand_dims(y2[:40], 1)

X_test = X2[40:]  # last 10 are testeing samples (20% of the data)
y_test = y2[40:]

print(len(X_train), len(X_test), len(y_train), len(y_test))

# visualize our data again

# plt.figure(figsize=(10, 7))
# plot test data in blue
# plt.scatter(X_train, y_train, c="b", label="Training data")
# Plot test data in green
# plt.scatter(X_test, y_test, c="g", label="Testing data")
# plt.legend()
# plt.show()

# lets build a neural network for our data

# 1 create model
model2 = tf.keras.Sequential([
    tf.keras.layers.Dense(50, input_shape=[1], name="input_layers"),
    tf.keras.layers.Dense(10, name="second_layer"),
    # The Last dense layer should have only 1 unit
    tf.keras.layers.Dense(1, name="output_layer")
], name="model_1")

model2.compile(loss=tf.keras.losses.mae,
               # *** Adam seems to work better than SGD as an optimizer for me
               optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
               metrics=["mae"])

# * total params - total number of parameters in the model.
# * Trainable parameters - these are the parameters (patterns) the model can update as it trains
# * Non-trainable params - these parameters aren't updated during training
# * (this is typical when you bring in already learned patterns or parameters
# from other models during **transfer learning**)
# ** exercise play around with number of hidden units in the dense layer

# 3. fit the model
# model2.fit(X_train, y_train, epochs=100, verbose=1)
# model2.summary()
#
# plot_model(model=model2, show_shapes=True, to_file="model.png")
# image = img.open("model.png")
# image.show()

# ** Visualizing our models predictions
# * To visualize predictions, its a good idea to plot them against the ground truth labels
# * Often you'll see this in the form of y_test, y_true versus y_pred (ground trueth versus your model)

# make some predictions
y_pred = model2.predict(X_test)


# y_pred = tf.reshape(y_prediction, [10])
# ** if you feel like your gonna reuse something turn it into a function
# Lets Create a plotting funtion

def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=y_pred):
    """Plots training data, test data and compares predictions to ground truth labels
    """
    plt.figure(figsize=(10, 7))
    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", label="Training Data")
    plt.scatter(test_data, test_labels, c="g", label="Test Data")
    plt.scatter(test_data, predictions, c="r", label="Predictions")
    plt.legend()
    plt.show(block=True)


# plot_predictions();

# ** Evaluating our model's predictions with regression evaluation metrics
# ** depending on the problem you're working on, there will be different evaluation metrics to
# evaluate your model's performance
# * Since we're working on a regression, two of the main metrics:
# * MAE - mean absolute error, "on average, how wrong is each of my model's predictions"
# * MSE - mean square error, "square the average errors" then find the average

# Evaluate the model on the test
model2.evaluate(X_test, y_test)


# Calculate the mean absolute error

# print("Line 268", tf.keras.losses.MAE(y_true=y_test,y_pred=tf.squeeze(y_pred)))

# calculate the mean square error

# print("Line 272", tf.keras.losses.MSE(y_true=y_test,y_pred=tf.squeeze(y_pred)))

# make some functions to reuse mae and mse

def mae(y_true, y_pred):
    return tf.keras.losses.MAE(y_true=y_true, y_pred=y_pred)


def mse(y_true, y_pred):
    return tf.keras.losses.MSE(y_true=y_true, y_pred=y_pred)


# Running experiments to improve our model


# 1) get more data - get more examples for your model to train on
# (more opportunities to learn patterns or relationships)
# 2) make your model larger ( using a more complex model) - this might come in the form of more layers
# or more hidden units in each layer
# 3 Train for longer - give you model more of a chance to find patterns in the data

# lets do 3 modelling experiments:

# 1. model_1 - same as the original model, 1 layer trained for 100 epochs
# 2 model_2 - 2layers, trained for 200 epochs
# 3 model_3  2 layers, trained for 500 epochs

# def model_1_training():
#     tf.random.set_seed(42)
modeltraining_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])
modeltraining_1.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.SGD(),
                        metrics=["mae"])
modeltraining_1.fit(X_train, y_train, epochs=100)
model_1_y_pred = modeltraining_1.predict(X_test)
# plot_predictions(predictions=model_1_y_pred)
mae_1 = mae(y_test, tf.squeeze(model_1_y_pred))
mse_1 = mse(y_test, tf.squeeze(model_1_y_pred))
print("mae: ", mae_1)
print("mse: ", mse_1)

# model_1_training()

# def model_2_training():
#     tf.random.set_seed(42)
modeltraining_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(7),
    tf.keras.layers.Dense(1)
])

modeltraining_2.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.SGD(),
                        metrics=["mse"])
modeltraining_2.fit(X_train, y_train, epochs=100)
model_2_y_pred = modeltraining_2.predict(X_test)
# plot_predictions(predictions=model_2_y_pred)
mae_2 = mae(y_test, tf.squeeze(model_2_y_pred))
mse_2 = mse(y_test, tf.squeeze(model_2_y_pred))
print("mae 2 : ", mae_2)
print("mse 2: ", mse_2)

# model_2_training()

# def model_3_training():
#     tf.random.set_seed(42)
modeltraining_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(50),
    tf.keras.layers.Dense(1)
])

modeltraining_3.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.SGD(),
                        metrics=["mse"])
modeltraining_3.fit(X_train, y_train, epochs=500)
model_3_y_pred = modeltraining_3.predict(X_test)
# plot_predictions(predictions=model_3_y_pred)
mae_3 = mae(y_test, tf.squeeze(model_3_y_pred))
mse_3 = mse(y_test, tf.squeeze(model_3_y_pred))
print("mae 3: ", mae_3)
print("mse 3: ", mse_3)

# model_3_training()

# Comparing the results of our experiments

# lets compare our models results using a pandas DataFrame
import pandas as pd

model_results = [["model_1", mae_1.numpy(), mse_1.numpy()],
                 ["model_2", mae_2.numpy(), mse_2.numpy()],
                 ["model_3", mae_3.numpy(), mse_3.numpy()]]

all_results = pd.DataFrame(model_results, columns=["model", "mae", "mse"])
# print(all_results)
# model 1 performed best?

# one of your main goals should be to minimize the time between your experiments you do the more things you'ss figure
# out which dont work and in turn, get closer to figuring out what does work. remember the michine leaning
# practioners motto experiment experiment experiment

## Tracking your experiments
# one really good habit in machine learning modelling is to track the reuslts of your experiments
# which can be tedious if your running lots of experiments
# luckily there are tools to help us!
# as you build more models, you'll want to look into using :
# TensorBoard a component of the tensorflow library to help track modelling experiments

# Weights and Biases - a tool for tracking all kinds of machine learning experiments
# plugs straight into tensorBoard

# Saving our models
# Savings our models allows us to use them outside of google colab or whereever they were trained
# such as a web application or a mobile app

# there are two main formats we can save our model's too:
#1 The savedModel format
#2 the HDF5 format

# Save the model using the savedModel format
# keras.src.saving.saving_api.save_model(modeltraining_3,"practice_saving_model_2.keras")
# Save model using the HDF5 format
# modeltraining_3.save("practice_saving_model.h5")

# Loading in a saved model format model
loaded_SavedModel_format = tf.keras.models.load_model("practice_saving_model_2.keras")

print(loaded_SavedModel_format.summary())
print(modeltraining_3.summary())

# Comparing modeltraining_3 with SavedModel format model predictions
modeltraining_3_preds = modeltraining_3.predict(X_test)
loaded_SavedModel_format_preds = loaded_SavedModel_format.predict(X_test)
# ** for whatever reason our data is different from when it gets saved,
# could be that were not setting a  seed

print(modeltraining_3_preds == loaded_SavedModel_format_preds)


# a larger example
# * Runnning experiments to improve our model

