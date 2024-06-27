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

# make 1000 examples
n_samples = 1000

# create circles
X, y = make_circles(n_samples, noise=0.03, random_state=42)

# Check out the features
print(X)

# Check out the labels
print(y[:10])

# * our data is a little hard to understand right now.. lets visulize
import pandas as pd
circles = pd.DataFrame({"X0": X[:, 0], "X1": X[:, 1], "label": y})
print(circles)

# visualize with a plot
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.show()


