# on my home computer i did all the gpu stuff

# import series of helper functions for the notebook
import wget
from helper_functions import unzip_data, create_tensorboard_callback, plot_loss_curves, compare_historys
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization

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

print(train_df_shuffled)
for row in train_df_shuffled[["text", "target"]][random_index:random_index + 20].itertuples():
    _, text, target = row
    print(f"Target: {target}", "(real disaster) " if target > 0 else "(not real disaster)")
    print(f"Text:\n {text} \n")
    print("---\n")

# splt data into rraining and validation sets

from sklearn.model_selection import train_test_split
# use train test split to split the training data into training and validations sets

train_sentences, val_sentences, train_labels, val_labels = train_test_split(train_df_shuffled["text"].to_numpy(),
                                                                            train_df_shuffled["target"].to_numpy(),
                                                                            test_size=0.1, #use 10% of training data for validation
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
print(round(sum([len(i.split()) for i in train_sentences])/len(train_sentences)))

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
print(text_vectorizer([sample_sentence]))


# Choose a random sentence from the training dataset and tokenize it
random_sentence = random.choice(train_sentences)
# print(f"Originial text: \n {random_sentence}\
#       \n\nVectorized version: ")
# print(text_vectorizer([random_sentence]))

# Get the unique words in the vocabulary
words_in_vocab = text_vectorizer.get_vocabulary()  # get all the unique words in our training data
top_5_words = words_in_vocab[:5]  # get the most common words
bottom_5_words = words_in_vocab[-5:]  # the least common words

print(f"Number of words in vocab: {len(words_in_vocab)}")
print(f"5 most common words: {top_5_words}")
print(f"5 least common words: {bottom_5_words}")

# creating an Embedding using an Embedding Layer
# to make our embedding, we're going to use tensorflow's embedding layer
# The parameters we care about for out embedding layers:
# input_dim** - the size of our vocabulary
# output_dim** - the size of the output embedding vector, for example, a value of
# 100 would mean each token gets represented by a vector 100 long
# input_length** - length of the sequence being passed to the embedding layer.

from tensorflow.keras import layers
embedding = layers.Embedding(input_dim=max_vocab_length,  # set input shape
                             output_dim=128)




