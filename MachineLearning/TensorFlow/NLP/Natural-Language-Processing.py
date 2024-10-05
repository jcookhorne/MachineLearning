# import series of helper functions for the notebook
import wget
# from helper_functions import unzip_data, create_tensorboard_callback, plot_loss_curves, compare_historys
import pandas as pd
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from tensorflow.keras.layers import TextVectorization

# url = 'https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/helper_functions.py'
# helper_functions = wget.download(url)


# get a text dataset

# The dataset we're going to be using is Kaggle's introduction to NLP dataset
# {text samples of Tweets labelled as disasters or not disaster

# skipped the unzip by just copying the data over here
# Become one with the data??
# Visualizing a text dataset
# to visualized we first have to read our datasets


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
# print(train_df.head())
# print(train_df["text"][0])
# print(train_df["text"][1])

# shuffle training dataframe
# good to shuffle data as long as data isn't sequential data
train_df_shuffled = train_df.sample(frac=1, random_state=42)
# print(train_df_shuffled.head())

# what does the test dataframe look like?
# print(test_df.head())

# how many examples of each class?
# print(train_df.target.value_counts())

# print(len(train_df), len(test_df))

# Let's visualize some random training examples
random_index = random.randint(0, len(train_df) - 5)  # create random indexes not higher than the total number of samples

# print(train_df_shuffled)
# for row in train_df_shuffled[["text", "target"]][random_index:random_index + 20].itertuples():
#     _, text, target = row
#     print(f"Target: {target}", "(real disaster) " if target > 0 else "(not real disaster)")
#     print(f"Text:\n {text} \n")
#     print("---\n")

# splt data into rraining and validation sets



# use train test split to split the training data into training and validations sets

train_sentences, val_sentences, train_labels, val_labels = train_test_split(train_df_shuffled["text"].to_numpy(),
                                                                            train_df_shuffled["target"].to_numpy(),
                                                                            test_size=0.1,
                                                                            #use 10% of training data for validation
                                                                            random_state=42)
# check teh lengths
# print(len(train_sentences), len(train_labels), len(val_sentences), len(val_labels))

# check the first 10 samples
# print(train_sentences[:10], train_labels[:10])

# converting text into numbers
# in NLP there are 2 main concepts for turning text into numbers
# Tokenization a straight mapping from word or character or sub-word to a numerical value
# Embeddings an embedding is a representation of natural language which can be learned
# *When dealing with a text problem, one of the first things you'll have to do before you can build a model is to
# convert your text to numbers.
# There are a few way to do this, namely:
# Tokenization - direct mapping of token (a token could be a word or a character) to number
# embedding - create a matrix of feature vector for each token (this size of the feature vector can be defined
# and this embedding can be learned)

# Text Vectorization (tokenization)
# print(train_sentences[:5])


# Find the average number of tokens (words) in the training tweets
# print(round(sum([len(i.split()) for i in train_sentences])/len(train_sentences)))

max_vocab_length = 10000  # max number of words to have in our vocabulary
max_length = 15  # max length our sequences will be (e.g. how many words from a Tweet does a model see?)

text_vectorizer = TextVectorization(max_tokens=max_vocab_length,  # how many words in the vocabulary
                                    standardize='lower_and_strip_punctuation',
                                    split="whitespace",
                                    ngrams=None,  # create groups of n-words
                                    output_mode="int",  # how to map tokens to numbers
                                    output_sequence_length=max_length,  # how long do you want your sequence to be?
                                    pad_to_max_tokens=True)

# Fit the text vectorizer to the training text
text_vectorizer.adapt(train_sentences)

# create a sample sentence and tokenize it
sample_sentence = "There is a flood on my street!"
text_vectorizer([sample_sentence])
# print(text_vectorizer([sample_sentence]))


# Choose a random sentence from the training dataset and tokenize it
random_sentence = random.choice(train_sentences)
# print(f"Originial text: \n {random_sentence}\
#       \n\nVectorized version: ")
# print(text_vectorizer([random_sentence]))

# Get the unique words in the vocabulary
words_in_vocab = text_vectorizer.get_vocabulary()  # get all the unique words in our training data
top_5_words = words_in_vocab[:5]  # get the most common words
bottom_5_words = words_in_vocab[-5:]  # the least common words

# print(f"Number of words in vocab: {len(words_in_vocab)}")
# print(f"5 most common words: {top_5_words}")
# print(f"5 least common words: {bottom_5_words}")

# creating an Embedding using an Embedding Layer
# to make our embedding, we're going to use tensorflow's embedding layer
# The parameters we care about for out embedding layers:
# input_dim** - the size of our vocabulary
# output_dim** - the size of the output embedding vector, for example, a value of
# 100 would mean each token gets represented by a vector 100 long
# input_length** - length of the sequence being passed to the embedding layer.

embedding = layers.Embedding(input_dim=max_vocab_length,  # set input shape
                             output_dim=128,  # output shape,
                             embeddings_initializer="uniform")  # how long is each input)
# print(embedding)

# Get a random sentence from the training set
# print(f"original text: \n {random_sentence} \n\nEmbedded version: ")
# Embed the random sentence (turn it into dense vectors of fixed size)
sample_embed = embedding(text_vectorizer([random_sentence]))
# print(f"{sample_embed}\n\n\n")

# check out a single token's embedding
# print(sample_embed[0][0])
# print(sample_embed[0][0].shape)
# print(random_sentence)


# ********** KEY THINGS
# the principles here are the text factorization or tokenization is converting words to some numerical
# format and then creating and embedding is making that just straight mapping, numerical format, making

# Modelli0ng a text dataset (running a series of experiments

# Now we've got a way to turn our text sequences into numbers, its time to start building
# a series of modelling experiments

# well start with a baseline and move on from there.
# good to look at scikit learn algorithms to build off of

# * Model 0: naive Bayes (baseline)
# * Model 1: Feed-forward neural network (dense model)
# * model 2: LSTM model (RNN)
# * model 3: GRU model (RNN)
# * model 4: Bidirectional-LSTM model (RNN)
# * model 5: 1D Convolutional neural network (CNN)
# * model 6: Tensorflow Hub Pretrained Feature Extractor (using transfer learning for NLP)
# * model 7: Same as model 6 with 10% of training data

# how are we going to approach all of these
# use the standard steps in modelling with tensorflow:
# Create a model
# Build a model
# Fit a model
# Evaluate a model

## MODEL 0 : getting a baseline
# as with all machine learning modelling experiments, it's important to create a baseline
# model so you've got a benchmark for future experiments to build on

# to create our baseline, we'll use sklearns multinomial naive bayes using the TF-IDF
# formula to convert our words to  numbers

# note its common practice to use non-deep learning algorithms as a baseline because of their speed and then
# later using deep learning to see if you can improve upon them



# Create tokenization and modelling pipeline
# pipeline is just it saying to do these things in this order
model_0 = Pipeline([
    ("tfidf", TfidfVectorizer()),  # convert words to number using tfidf
    ("clf", MultinomialNB())  # model the text
])
# clf is short for classifier

# fit the pipeline to the training data
model_0.fit(train_sentences, train_labels)

# evaluate our baseline model
baseline_score = model_0.score(val_sentences, val_labels)

print(f"our baseline model achieves an accuracy of: {baseline_score* 100:.2f}%")


# print(train_df)
print(train_df.target.value_counts())

# make predictions

baseline_preds = model_0.predict(val_sentences)
print(baseline_preds[:20])


# Function to evaluate: accuracy, precision, recall, f1-score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_results(y_true, y_pred):
    """
    Calculates model accuracy, precision, recall and f1-score score of a binary classification model

    """
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    # Calculate model precision, recall and f1-score using "weighted" average
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = {"accuracy": model_accuracy,
                     "precision": model_precision,
                     "recall": model_recall,
                     "f1": model_f1}
    return model_results


# Get baseline results
baseline_results = calculate_results(y_true=val_labels,
                                     y_pred=baseline_preds)

print(f"these are the baseline results: {baseline_results}")

# *** MODEL 1: A Simple Dense Model

# Create a tensorboard callback( need to create a new one for each model)
from helper_functions import create_tensorboard_callback

# Create a directory to save TensorBoard logs
SAVE_DIR = "model_logs"

# Build model with the functional API
# *Functional api is more customizable than the sequential api
from tensorflow.keras import layers
inputs = layers.Input(shape=(1,), dtype=tf.string)  # inputs are 1-dimensional strings
test = text_vectorizer(inputs)  # turn the input text into numbers
test = embedding(test)  # create an embedding of the numberized inputs
print(test.shape)
# our input are 1 dimensional but our output is not one dimensional?
test = tf.keras.layers.GlobalAveragePooling1D()(test) # lower the dimensionality of the embedding to 1 dimension
outputs = layers.Dense(1, activation="sigmoid")(test)  # create the output layer, want the binary outputs so use
# sigmoid activation functions
model_1 = tf.keras.Model(inputs, outputs, name="model_1_dense")
print(f"summary of the model: {model_1.summary()}")

#  # there is something wrong with the shape of the dense layer

# Compile model
model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])
# Fit the model_1
# model_1_history = model_1.fit(x=train_sentences,
#                               y=train_labels,
#                               epochs=5,
#                               validation_data=(val_sentences, val_labels),
#                               callbacks=[create_tensorboard_callback(dir_name=SAVE_DIR,
#                                                                      experiment_name="model_1_dense")])

# make some prediction and evaluate those
model_1_prediction_probability = model_1.predict(val_sentences)
print(model_1_prediction_probability[:10])

# convert model prediction probabilities to label format
model_1_preds = tf.squeeze(tf.round(model_1_prediction_probability))

print(model_1_preds[:20])

# Calculate our model_1 results
model_1_results = calculate_results(y_true=val_labels,
                                    y_pred=model_1_preds)
print(f"this is the results of model_1: {model_1_results}")

import numpy as np
print(np.array(list(model_1_results.values())) > np.array(list(baseline_results.values())))

