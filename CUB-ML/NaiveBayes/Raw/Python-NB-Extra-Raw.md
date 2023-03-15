---
layout: page
title: "Gates NB-Extra Raw-code-Python"
permalink: /CUB-ML/Raw/NB-Extra-Python
---

# Naive Bayes - Extra code (in Python) - Dr. Gates' full, raw (unedited) code
Reference: Professor Ami Gates, Dept. Applied Math, Data Science, University of Colorado

[Dr. Gates' Website](https://gatesboltonanalytics.com/)

```python
import nltk
import pandas as pd
import sklearn
import numpy as np
from sklearn import preprocessing

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
## Read in clean data for MN NB
filename="C:/Users/profa/Desktop/UCB/ML CSCI 5622/HeartRisk_JustNums_3_labels.csv"
DF1=pd.read_csv(filename)
print(DF1)

## Create test and train sets and remove labels

#TrainDF, TestDF, TrainLabels, TestLabels = 
TrainDF, TestDF = train_test_split(DF1, test_size=0.33)
print(TrainDF)
print(TestDF)

TrainLabels = TrainDF["Label"]
TestLabels = TestDF["Label"]

TrainDF.drop(columns=['Label'])
TestDF.drop(columns=['Label'])


# ## Normalize the data
# DataPart=DF1.iloc[:,1:4]

# ColumnNames=DataPart.columns
# print(ColumnNames)

# DF_norm = preprocessing.normalize(DataPart, axis=0)  ##axis=0 is column wise
# DF_norm = pd.DataFrame(DF_norm, columns=ColumnNames)
# DF_norm.head()

################################################################
## Multinomial NB ----------------
###################################################################

# https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
## Instantiate the model
## alpha = 1 is for Laplace smoothing
MyModelNB= MultinomialNB(alpha=1)

## Run the model using fit
MyModelNB.fit(TrainDF, TrainLabels)
##Use the model to predict the testset
Prediction1 = MyModelNB.predict(TestDF)

print("\nThe prediction from NB is:")
print(Prediction1)
print("\nThe actual labels are:")
print(TestLabels)


## confusion matrix

## The confusion matrix is square and is labels X labels
## We have two labels, so ours will be 2X2
#The matrix shows
## rows are the true labels
## columns are predicted
## it is alphabetical
## The numbers are how many 
cnf_matrix1 = confusion_matrix(TestLabels, Prediction1)
print("\nThe confusion matrix is:")
print(cnf_matrix1)

### prediction probabilities
## columns are the labels in alphabetical order
## The decimal in the matrix are the prob of being
## that label
print(np.round(MyModelNB.predict_proba(TestDF),2))
```
