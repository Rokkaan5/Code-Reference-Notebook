# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:50:30 2022

@author: profa
"""
import numpy as np
import nltk
import pandas as pd
import sklearn
import re  
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import os

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string

from bs4 import BeautifulSoup
from collections import Counter

from nltk.corpus import stopwords
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize

################
##
## ANN - CNN - RNN
## Movies Dataset
## https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format
##
####  Gates
##################################################################################
path = "C:/Users/profa/Desktop/UCB/NNCSCI5922/Code/IMDB_Movie_Datasets_Train_Test_Valid/"
TrainData = pd.read_csv(str(path+"Train.csv"))
# print(TrainData.shape)
print(TrainData.head(10))
# print(type(TrainData))

TestData = pd.read_csv(str(path+"Test.csv"))
#print(TestData.shape)
TestData.head(10)

ValidData = pd.read_csv(str(path+"Valid.csv"))
#print(ValidData.shape)
ValidData.head(10)

## Concat requires a list
## Place all data from above into one dataframe
FullDataset=pd.concat([TrainData,TestData, ValidData])
#print(FullDataset.shape)

###################################################
## The following code represents a more
## hands-on option for tokenizing/vectorizing
## the data. I am leaving it here as a reference
##
## BELOW this comment area - CountVectorizer is used
## to perform the same tasks. 
###########################################################

# def remove_html(text):
#     bs = BeautifulSoup(text, "html.parser")
#     return ' ' + bs.get_text() + ' '
 
# def keep_only_letters(text):
#     text=re.sub(r'[^a-zA-Z\s]',' ',text)
#     return text
 
# def convert_to_lowercase(text):
#     return text.lower()

# def remove_small_words(text):
#     text=" ".join(word for word in text.split() if len(word)>=3)
#     return text
    
    
# def clean_reviews(text):
#     text = remove_html(text)
#     text = keep_only_letters(text)
#     text = convert_to_lowercase(text)
#     text = remove_small_words(text)
#     return text

# def returnNewDF(oldDF):
#     newDF=pd.DataFrame(columns=["key", "value"])
#     if not oldDF["key"] in stopwords.words():
#         newDF["key"] = oldDF["value"]
#     return newDF


 
# TrainData["text"] = TrainData["text"].apply(lambda text: clean_reviews(text))
# print(TrainData.head(30))

# TestData["text"] = TestData["text"].apply(lambda text: clean_reviews(text))
# print(TestData.head(30))

# ValidData["text"] = ValidData["text"].apply(lambda text: clean_reviews(text))
# print(ValidData.head(30))


# ## Create Vocab
# counter = Counter([words for reviews in TrainData["text"] for words in reviews.split()])
# df = pd.DataFrame()
# df['key'] = counter.keys()
# df['value'] = counter.values()
# df.sort_values(by='value', ascending=False, inplace=True)
# print(df.head(10))

# ## Drop all the stopwords - OPTIONAL - .............
# #df = df[~df.key.isin(stopwords.words())]
# #print(stopwords.words())

####################################################################

## Clean Up TrainData
## Get the vocab

#print(TrainData.head())
# Testing iterating the columns 
for col in TrainData.columns: 
    print(col) 
    
## Check Content   --------------------
#print(TrainData["text"])
#print(TrainData["label"]) ##0 is negative, 1 is positive

### Tokenize and Vectorize 
## Create the list 
## Keep the labels

ReviewsLIST=[]  ## from the text column
LabelLIST=[]    

for nextreview, nextlabel in zip(TrainData["text"], TrainData["label"]):
    ReviewsLIST.append(nextreview)
    LabelLIST.append(nextlabel)

print("A Look at some of the reviews list is:\n")
print(ReviewsLIST[0:20])

print("A Look at some of the labels list is:\n")
print(LabelLIST[0:20])


######################################
## Optional - for Stemming the data
##
################################################
## Instantiate it
A_STEMMER=PorterStemmer()
## test it
print(A_STEMMER.stem("fishers"))
#----------------------------------------
# Use NLTK's PorterStemmer in a function - DEFINE THE FUNCTION
#-------------------------------------------------------
def MY_STEMMER(str_input):
    ## Only use letters, no punct, no nums, make lowercase...
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [A_STEMMER.stem(word) for word in words] ## Use the Stemmer...
    return words

#########################################
##
##  Build the labeled dataframe
##  Get the Vocab  - here keeping top 10,000
##
######################################################

### Vectorize
## Instantiate your CV
MyCountV=CountVectorizer(
        input="content",  
        lowercase=True, 
        #stop_words = "english", ## This is optional
        #tokenizer=MY_STEMMER, ## Stemming is optional
        max_features=11000  ## This can be updated
        )

## Use your CV 
MyDTM = MyCountV.fit_transform(ReviewsLIST)  # create a sparse matrix
#print(type(MyDTM))


ColumnNames=MyCountV.get_feature_names() ## This is the vocab
#print(ColumnNames)
#print(type(ColumnNames))

## Here we can clean up the columns


## Build the data frame
MyDTM_DF=pd.DataFrame(MyDTM.toarray(),columns=ColumnNames)

## Convert the labels from list to df
Labels_DF = pd.DataFrame(LabelLIST,columns=['LABEL'])

## Check your new DF and you new Labels df:
# print("Labels\n")
print(Labels_DF)
# print("DF\n")
print(MyDTM_DF.iloc[:,0:20])
print(MyDTM_DF.shape) ## 40,000 by 11000

############################################
##
##  Remove any columns that contain numbers
##  Remove columns with words not the size 
##  you want. For example, words<3 chars
##
##############################################
##------------------------------------------------------
### DEFINE A FUNCTION that returns True if numbers
##  are in a string 
def Logical_Numbers_Present(anyString):
    return any(char.isdigit() for char in anyString)
##----------------------------------------------------

for nextcol in MyDTM_DF.columns:
    #print(nextcol)
    ## Remove unwanted columns
    #Result=str.isdigit(nextcol) ## Fast way to check numbers
    #print(Result)
    
    ##-------------call the function -------
    LogResult=Logical_Numbers_Present(nextcol)
    #print(LogResult)
    ## The above returns a logical of True or False
    
    ## The following will remove all columns that contains numbers
    if(LogResult==True):
        #print(LogResult)
        #print(nextcol)
        MyDTM_DF=MyDTM_DF.drop([nextcol], axis=1)

    ## The following will remove any column with name
    ## of 3 or smaller - like "it" or "of" or "pre".
    ##print(len(nextcol))  ## check it first
    ## NOTE: You can also use this code to CONTROL
    ## the words in the columns. For example - you can
    ## have only words between lengths 5 and 9. 
    ## In this case, we remove columns with words <= 3.
    elif(len(str(nextcol))<3):
        print(nextcol)
        MyDTM_DF=MyDTM_DF.drop([nextcol], axis=1)
       
    

##Save original DF - without the lables
My_Orig_DF=MyDTM_DF
#print(My_Orig_DF)

 

## Now - let's create a complete and labeled
## dataframe:
dfs = [Labels_DF, MyDTM_DF]
print(dfs)
print("shape of labels\n", Labels_DF)
print("shape of data\n", MyDTM_DF)

Final_DF_Labeled = pd.concat(dfs,axis=1, join='inner')
## DF with labels
print(Final_DF_Labeled.iloc[:, 0:2])
print(Final_DF_Labeled.shape)


################################################
## FYI
## An alternative option for most frequent 10,000 words 
## Not needed here as we used CountVectorizer with option
## max_features
# print (df.shape[0])
# print (df[:10000].value.sum()/df.value.sum())
# top_words = list(df[:10000].key.values)
# print(top_words)
# ## Example using index
# index = top_words.index("humiliating")
# print(index)
##############################################

## Create list of all words
print(Final_DF_Labeled.columns[0])
NumCols=Final_DF_Labeled.shape[1]
print(NumCols)
print(len(list(Final_DF_Labeled.columns)))

top_words=list(Final_DF_Labeled.columns[1:NumCols+1])
## Exclude the Label

print(top_words[0])
print(top_words[-1])


print(type(top_words))
print(top_words.index("aamir")) ## index 0 in top_words
print(top_words.index("zucco")) #index NumCols - 2 in top_words

## Encoding the data
def Encode(review):
    words = review.split()
   # print(words)
    if len(words) > 500:
        words = words[:500]
        #print(words)
    encoding = []
    for word in words:
        try:
            index = top_words.index(word)
        except:
            index = (NumCols - 1)
        encoding.append(index)
    while len(encoding) < 500:
        encoding.append(NumCols)
    return encoding
##-------------------------------------------------------
## Test the code to assure that it is
## doing what you think it should 

result1 = Encode("aaron aamir abbey abbott abilities zucco ")
print(result1)
result2 = Encode("york young younger youngest youngsters youth youthful youtube zach zane zany zealand zellweger")
print(result2)
print(len(result2)) ## Will be 500 because we set it that way above
##-----------------------------------------------------------
 
###################################
## Now we are ready to encode all of our
## reviews - which are called "text" in
## our dataset. 

# Using vocab from above i -  convert reviews (text) into numerical form 
# Replacing each word with its corresponding integer index value from the 
# vocabulary. Words not in the vocab will
# be assigned  as the max length of the vocab + 1 
## ########################################################

# Encode our training and testing datasets
# with same vocab. 

print(TestData.head(10))
print(TestData.shape)
print(TrainData.shape)


############### Final Training and Testing data and labels-----------------
training_data = np.array([Encode(review) for review in TrainData["text"]])
print(training_data[20])
print(training_data.shape)

testing_data = np.array([Encode(review) for review in TestData['text']])
print(testing_data[20])

validation_data = np.array([Encode(review) for review in ValidData['text']])

print (training_data.shape, testing_data.shape)

## Prepare the labels if they are not already 0 and 1. In our case they are
## so these lines are commented out and just FYI
#train_labels = [1 if label=='positive' else 0 for sentiment in TrainData['label']]
#test_labels = [1 if label=='positive' else 0 for sentiment in TestData['label']]
train_labels = np.array([TrainData['label']])
train_labels=train_labels.T
print(train_labels.shape)
test_labels = np.array([TestData['label']])
test_labels=test_labels.T
print(test_labels.shape)

###############################
## ANN
#################################
## Simple Dense NN for sentiment analysis (classification 0 neg, 1 pos)
# First layer: Embedding Layer (Keras Embedding Layer) that will learn embeddings 
# for different words .
## RE: ## https://keras.io/api/layers/core_layers/embedding/
## input_dim: Integer. Size of the vocabulary
## input_length: Length of input sequences, when it is constant.
print(NumCols)   
input_dim = NumCols + 1 
import tensorflow
from tensorflow.keras.layers import Activation
 #https://www.tensorflow.org/api_docs/python/tf/keras/Input
input_data = tensorflow.keras.layers.Input(shape=(500))
 #https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding

data = tensorflow.keras.layers.Embedding(input_dim=input_dim, output_dim=64, input_length=500)(input_data)
##input_dim: Integer. Size of the vocabulary, i.e. maximum integer index + 1
## Good tutorial for this concept:
    ## https://medium.com/analytics-vidhya/understanding-embedding-layer-in-keras-bbe3ff1327ce
 #output_dim: Integer. Dimension of the dense embedding.
 # output_dim: This is the size of the vector space in which words will be embedded. 
 #It defines the size of the output vectors from this layer for each word. 
 # For example, it could be 32 or 100 or even larger.
 #https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
#  In an embedding, words are represented by dense vectors where a vector represents 
#  the projection of the word into a continuous vector space.
# The position of a word within the vector space is learned 
# from text and is based on the words that surround the word when it is used.
# The position of a word in the learned vector space is referred to as its embedding.
# data = tensorflow.keras.layers.Flatten()(data)
 #Dense layers require inputs as (batch_size, input_size) 
data = tensorflow.keras.layers.Dense(16)(data)
data = tensorflow.keras.layers.Activation('relu')(data)
#data = tensorflow.keras.layers.Dropout(0.5)(data)
 
data = tensorflow.keras.layers.Dense(8)(data)
data = tensorflow.keras.layers.Activation('relu')(data)

#data = tensorflow.keras.layers.Dropout(0.5)(data)
 
data = tensorflow.keras.layers.Dense(4)(data)
data = tensorflow.keras.layers.Activation('sigmoid')(data)
#data = tensorflow.keras.layers.Dropout(0.5)(data)
 
data = tensorflow.keras.layers.Dense(1)(data)
output_data = tensorflow.keras.layers.Activation('sigmoid')(data)
 
model = tensorflow.keras.models.Model(inputs=input_data, outputs=output_data)
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')
model.summary()

print(training_data[0:3, 0:3])
print(training_data.shape)
model.fit(training_data, train_labels, epochs=10, batch_size=256, validation_data=(testing_data, test_labels))


###################################
## RNN
###############################################

import tensorflow
from tensorflow.keras.layers import Activation
 
input_data = tensorflow.keras.layers.Input(shape=(500))
 
data = tensorflow.keras.layers.Embedding(input_dim=input_dim, output_dim=32, input_length=500)(input_data)
 #https://www.tensorflow.org/api_docs/python/tf/keras/layers/Bidirectional
data = tensorflow.keras.layers.Bidirectional(tensorflow.keras.layers.SimpleRNN(50))(data)
 
data = tensorflow.keras.layers.Dense(1)(data)
output_data = tensorflow.keras.layers.Activation('sigmoid')(data)
 
model = tensorflow.keras.models.Model(inputs=input_data, outputs=output_data)
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')
model.summary()

model.fit(training_data, train_labels, epochs=10, batch_size=256, validation_data=(testing_data, test_labels))


############################################
## LSTM
#############################################
import tensorflow
from tensorflow.keras.layers import Activation
 
input_data = tensorflow.keras.layers.Input(shape=(500))
 
data = tensorflow.keras.layers.Embedding(input_dim=input_dim, output_dim=32, input_length=500)(input_data)
 
data = tensorflow.keras.layers.Bidirectional(tensorflow.keras.layers.LSTM(50))(data)
 
data = tensorflow.keras.layers.Dense(1)(data)
output_data = tensorflow.keras.layers.Activation('sigmoid')(data)
 
model = tensorflow.keras.models.Model(inputs=input_data, outputs=output_data)
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')
model.summary()

model.fit(training_data, train_labels, epochs=10, batch_size=128, validation_data=(testing_data, test_labels))

 

 

######################################
## CNN
########################################
import tensorflow
 
input_data = tensorflow.keras.layers.Input(shape=(500))
 
data = tensorflow.keras.layers.Embedding(input_dim=input_dim, output_dim=32, input_length=500)(input_data)
 
data = tensorflow.keras.layers.Conv1D(50, kernel_size=3, activation='relu')(data)
data = tensorflow.keras.layers.MaxPool1D(pool_size=2)(data)
 
data = tensorflow.keras.layers.Conv1D(40, kernel_size=3, activation='relu')(data)
data = tensorflow.keras.layers.MaxPool1D(pool_size=2)(data)
 
data = tensorflow.keras.layers.Conv1D(30, kernel_size=3, activation='relu')(data)
data = tensorflow.keras.layers.MaxPool1D(pool_size=2)(data)
 
data = tensorflow.keras.layers.Conv1D(30, kernel_size=3, activation='relu')(data)
data = tensorflow.keras.layers.MaxPool1D(pool_size=2)(data)
 
data = tensorflow.keras.layers.Flatten()(data)
 
data = tensorflow.keras.layers.Dense(20)(data)
data = tensorflow.keras.layers.Dropout(0.5)(data)
 
data = tensorflow.keras.layers.Dense(1)(data)
output_data = tensorflow.keras.layers.Activation('sigmoid')(data)
 
model = tensorflow.keras.models.Model(inputs=input_data, outputs=output_data)
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')
model.summary()

model.fit(training_data, train_labels, epochs=10, batch_size=256, validation_data=(testing_data, test_labels))


print("Evaluate model on test data")
results = model.evaluate(testing_data, test_labels, batch_size=256)
print("test loss, test acc:", results)

# Generate a prediction using model.predict() 
# and calculate it's shape:
print("Generate a prediction")
prediction = model.predict(testing_data)
print(prediction)
print("prediction shape:", prediction.shape)
print(type(prediction))
prediction[prediction > .5] = 1
prediction[prediction <= .5] = 0
print(prediction)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(prediction, test_labels))