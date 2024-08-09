
# on my home computer i did all the gpu stuff

# import series of helper functions for the notebook
import wget
from helper_functions import unzip_data, create_tensorboard_callback, plot_loss_curves, compare_historys
# url = 'https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/helper_functions.py'
# helper_functions = wget.download(url)
#

# get a text dataset

# The dataset we're going to be using is Kaggle's introduction to NLP dataset
# {text samples of Tweets labelled as disasters or not disaster

# skipped the unzip by just copying the data over here


# Become one with the data??
# Visualizing a text dataset
# to visualized we first have to read our datasets

import pandas as pd
train_df = pd.read_csv("train.csv")



