# Examples of computer vision problems

"""
Binary classification is this a picture of one thing or another

Multiclass classification is this a photo of more than one thing or another

Object detection where is the thing we're looking for?

*** WHat we're going to cover broadly **
* getting a dataset to work with images of food
* Architecture of a convulutional neural network (CNN) with tensorflow
* an end-to-end  binary image classification problem
* Steps in modelling with CNNs
  * Creating a CNN, compiling a model, fitting a model, evaluating a model
* An end-to-end multi-class image classification problem
* making predictions on our own custom images

        HOW: bny cooking code and experimenting a bunch
        
        CNN are commonly used to solve problems involving spatial data such as images
        ** - data that has no direct correlation to each other
        Rnn  are better suited to analyzing sequential data such as text or videos
        ** - data that usually have some relation to each other or order
"""

# introduction to Convolutional Neural networks and Computer Vision with Tensorflow

# Computer Vision is the practice of writing algorithms which can discover patterns in visual data.
# such as the camera of a self-driving car recognizing the car in front

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import tensorflow as tf
import numpy as np
import zipfile
import wget
import os
import pathlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Commented so we dont keep downloading it
# # First get the data (from food 101 at kaggle
# wget.download('https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip')
# # Unzip the downloaded file
# zip_ref = zipfile.ZipFile("pizza_steak.zip")
# zip_ref.extractall()
# zip_ref.close()


# the images we're working witha re from the food101 dataset (101 different classes of food): kaggle

# however we've modified it to only use two classes ( pizza & steak) using the image
# data modification notebook this is something he did before we downloaded it

# start smaller so we can experiment quickly and figure out what works and what doesn't before scaling

# Inspect the data ( Become one with it)
# A very crucial step at the beginning of any machine learning project is becoming one with the data
# and for a computer vision project ... this usually means visualizing many samples of our data
# walk through pizza_steak directory and list number of files
for dirpath, dirnames, filenames in os.walk("pizza_steak"):
    print(f"there are {len(dirnames)} directroies and {len(filenames)}  images in {dirpath}.")

# another way to find out how many images are in a file
num_steak_images_train = len(os.listdir("pizza_steak/train/steak"))
print(num_steak_images_train)

# To visualize our images, first lets get teh class names programmatically
# get the classnames programmatically
data_dir = pathlib.Path("pizza_steak/train")
# created a list of classnames from subdirectories
class_names = np.array(sorted([item.name for item in data_dir.glob("*")]))
print(class_names)

# lets visualize our images

# Setup the target directory (we'll view images from here)
def view_random_image(target_dir, target_class):
    target_folder = target_dir+target_class
    # get a random image path
    random_image = random.sample(os.listdir(target_folder), 1)
    print(random_image)
    # Read in the image and plot it using matplotlib
    img = mpimg.imread(target_folder + "/" + random_image[0])
    plt.imshow(img)
    plt.title(target_class)
    plt.axis("off")
    plt.show()
    # print(f"Image shape: {img.shape}")  # show the shape of the image
    return img


# view a random image from the training dataset
img = view_random_image(target_dir="pizza_steak/train/",
                        target_class="pizza")

# the images we've imported and plotted are actually giant arrays/tensors of different pixel values
# print(tf.constant(img))

# View the image shape
print("IMAGE SHAPE : ",img.shape) # returns width, height, color channels

# Get all the pixel values between 0 & 1
print("PIXEL VALUES: ",img/255)

# * Load our images
# * Preprocess our images
# * Build a CNN to find patterns in our images
# * Compile our CNN
# * Fit the CNN to our training data


# set the seed
tf.random.set_seed(42)

# Preprocess data (get all of the pixels)


