# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 20:03:53 2018
@author: iit
Can you categorize news articles based on their headlines and short descriptions?


"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Conver json file data into a proper json data
#first way of reading the json file
'''
import json
with open("News_Category_Dataset.json") as datafile:
    data = json.load(datafile)
dataframe = pd.DataFrame(data)
'''
# 2 nd way of reading the file
dataset = pd.read_json('News_Category_Dataset.json' )
no_of_cateogories=list(set(list(dataset.iloc[:,1])))
#cleaning the text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus_headline=[]
corpus_description=[]
for i in range(0,124989):
    headline = re.sub('[^a-zA-Z]',' ',dataset['headline'][i])
    headline = headline.lower()
    headline = headline.split()
    ps = PorterStemmer()
    headline = [ps.stem(word) for word in headline if not word in set(stopwords.words('english'))]
    headline = ' '.join(headline)
    corpus_headline.append(headline)
    description = re.sub('[^a-zA-Z]',' ',dataset['short_description'][i])
    description = description.lower()
    description = description.split()
    ps = PorterStemmer()
    description = [ps.stem(word) for word in description if not word in set(stopwords.words('english'))]
    description = ' '.join(description)
    corpus_description.append(description)
corpus = {'headline':corpus_headline , 'short_description':corpus_description}
X = pd.DataFrame(corpus)
y = dataset.iloc[:, 1].values
# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y).reshape(-1, 1)
onehotencoder = OneHotEncoder(categorical_features = [0])
y = onehotencoder.fit_transform(y).toarray()

# Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus_description).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy' ,random_state=0)
classifier.fit(X_train , y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))