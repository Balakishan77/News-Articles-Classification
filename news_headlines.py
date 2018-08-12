# This Python 3 environment comes with many helpful analytics libraries installed
#Categorizing news articles based on their headlines and short descriptions
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Importing the dataset
dataset = pd.read_json('News_Category_Dataset.json', lines='True') #Reading the file as a json object per line.
dataset .columns #Index(['authors', 'category', 'date', 'headline', 'link', 'short_description'], dtype='object')
dataset.shape  #(124989, 6)
#Checking for duplicates and removing them
dataset.drop_duplicates(inplace = True)
dataset.shape  #(124986, 6)
#Checking for any null entries in the dataset
print (pd.DataFrame(dataset.isnull().sum()))
'''
authors            0
category           0
date               0
headline           0
link               0
short_description  0
'''
#Both WORLDPOST and THE WORLDPOST are same, so will merge them into one
dataset.category = dataset.category.map(lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)
no_of_cateogories=list(set(list(dataset.iloc[:,1])))
#Total of 31 cateogiries
categories=(dataset.groupby('category')).size()
'''
number of data samples by categories
ARTS               1509
ARTS & CULTURE     1339
BLACK VOICES       3858
BUSINESS           4254
COLLEGE            1144
COMEDY             3971
CRIME              2893
EDUCATION          1004
ENTERTAINMENT     14257
FIFTY              1401
GOOD NEWS          1398
GREEN              2622
HEALTHY LIVING     6694
IMPACT             2602
LATINO VOICES      1129
MEDIA              2815
PARENTS            3955
POLITICS          32739
QUEER VOICES       4995
RELIGION           2556
SCIENCE            1381
SPORTS             4167
STYLE              2254
TASTE              2096
TECH               1231
THE WORLDPOST      3664
TRAVEL             2145
WEIRD NEWS         2670
WOMEN              3490
WORLD NEWS         2177
WORLDPOST          2579
dtype: int64
'''
#Using Natural Language Processing to cleaning the text and combining headlines and short_descriptions to make one corpus
import re
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
dataset['headline_description'] = dataset['headline'] + " " + dataset['short_description']
corpus=[]
invalid_data=[]
headline_description_corpus=[]
for i in range(0,len(dataset['headline_description'])):
    try:
        headline_description_corpus = re.sub('[^a-zA-Z]',' ',str(dataset['headline_description'][i]))
    except Exception as e:
        invalid_data.append(i)
        print (e,i)
        continue
    headline_description_corpus = headline_description_corpus.lower()
    headline_description_corpus = headline_description_corpus.split()
    ps = PorterStemmer()
    headline_description_corpus = [ps.stem(word) for word in headline_description_corpus if not word in set(stopwords.words('english'))]
    headline_description_corpus = ' '.join(headline_description_corpus)
    corpus.append(headline_description_corpus)
    
# Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
X_1 = cv.fit_transform(corpus).toarray()
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)
X = transformer.fit_transform(X_1).toarray()
# Encoding the Dependent Variable
y = dataset.iloc[:, 1].values
y=np.delete(y, invalid_data)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y).reshape(-1, 1)
onehotencoder = OneHotEncoder(categorical_features = [0])
y = onehotencoder.fit_transform(y).toarray()
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
'''
# Fitting classifier to the Training set
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
y_train = np.argmax(y_train, axis=1)
classifier.fit(X_train , y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

y_test = np.argmax(y_test, axis=1)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#this function computes subset accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred) #0.336427062664021
accuracy_score(y_test, y_pred,normalize=False) #10512

'''

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy' ,random_state=0)
classifier.fit(X_train , y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

y_test = np.argmax(y_test, axis=1)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#this function computes subset accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred) #0.336427062664021
accuracy_score(y_test, y_pred,normalize=False) #10512
