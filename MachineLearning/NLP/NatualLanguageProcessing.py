import numpy as np
import pandas as pd
import matplotlib as plt

# the delimeter parameter helps to specify it is  tsv file
# the quoting parameter when set to 3 allows you to ignore quotes
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3 )


#Cleaning the Text
#Stemming removes extra parts of words that would have the same meaning
#such as loved to love
#Stop words are like articles meaning the is are empty words
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    #re.sub will replace anything in the first parameter by the second
    #so anything that is not an upper or lowercase letter into a space
    #third parameter is choosing the dataset and the specific Column
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower() #review will only have lowercase values now
    review = review.split() #split will split them into different words
    ps = PorterStemmer()
    allStopwords = stopwords.words('english')
    allStopwords.remove('not')
    allStopwords.remove("isn't")
    review = [ps.stem(word)for word in review if not word in set(allStopwords)]
    review = ' '.join(review)
    corpus.append(review)
# print(*corpus, sep="\n")

#Creating the bag of words
from sklearn.feature_extraction.text import CountVectorizer
#Max features will only take the most frequently used words up to the amount given
#so it will only take the most frequently used 1500 words
cv = CountVectorizer(max_features=1500)
#Fit will take all the words and tranform method will put all of them in different
# columns

#the dependent variable will always be y

x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

#Finds the number of words in the first row or number of columns of x
# print(len(x[0]))
# print(x)
# print(y)

# TRAINING SET
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#TRAINING NAIVE BAYES MODEL
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train);

#predicting the test set result
y_pred = classifier.predict(x_test)
# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

#MAkING THE CONFUSION MATRIX
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))


