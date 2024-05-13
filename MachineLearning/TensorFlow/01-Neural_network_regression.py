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
newModel.compile(loss="mae",
                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                 metrics=["mae"])

# 3 fit the model
newModel.fit(tf.expand_dims(X1, axis=-1), tf.expand_dims(y1, axis=-1), epochs=100)

print(X,'\n', y)
print(newModel.predict(x=tf.cast([17.0], dtype=tf.float32)))

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

print(X2, y2)
# plt.plot(X2, y2)
plt.scatter(X2, y2)
plt.show()

# * the 3 sets . . .

# * Training set - the model learns from this data, which is typically 70-80% of the total data you have available.\
# * Validation set - the model gets tuned on this data, which is typically 10-15% of the data available.
#  * Test set - the model gets evaluated on this data to test what is has learned, this set is typically 10-15% of
#  the total data available.
# * you don't always need the validation set







