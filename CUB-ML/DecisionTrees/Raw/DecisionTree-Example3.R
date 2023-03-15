################################################
## 
##  Basic R Decision Trees Tutorial for 
##  Mixed data types on a CLEAN record dataset
##  and a numeric text corpus
##
##  Gates
################################################
## The record data used is here:
##  https://drive.google.com/file/d/18dJPOiiO9ogqOibJppc0lsDiQ2-bQs0f/view?usp=sharing
## The text corpus used is here:
##  https://drive.google.com/drive/folders/1NnPZfhg_04lUAdKPNMMqkTcyfF6Z0BP2?usp=sharing
##
################################################

library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(Cairo)
library(network)
##If you install from the source....
#Sys.setenv(NOAWT=TRUE)
## ONCE: install.packages("wordcloud")
library(wordcloud)
## ONCE: install.packages("tm")
library(tm)
# ONCE: install.packages("Snowball")
## NOTE Snowball is not yet available for R v 3.5.x
## So I cannot use it  - yet...
##library("Snowball")
##set working directory
## ONCE: install.packages("slam")
library(slam)
library(quanteda)
## ONCE: install.packages("quanteda")
## Note - this includes SnowballC
library(SnowballC)
library(arules)
##ONCE: install.packages('proxy')
library(proxy)
## ONCE: if needed:  install.packages("stringr")
library(stringr)
## ONCE: install.packages("textmineR")
library(textmineR)
library(igraph)
library(lsa)


## These are paths on MY COMPUTER - not yours :)
RecordDataPath="C:/Users/profa/Documents/Python Scripts/ANLY503/DATA/StudentSummerProgramDataClean_DT.csv"
TextCorpusPath="C:/Users/profa/Documents/Python Scripts/TextMining/DATA/Sent_Corpus"
#######################################
##  Read in and prepare the data.
######################################################
##  The record data is easy - it already has a label and is clean...
RecordDF=read.csv(RecordDataPath, )
head(RecordDF)
##-------------------------------------------------------
## Read in and prep the text data from the corpus
(NCorpus <- Corpus(DirSource(TextCorpusPath)))
## Here, I am keeping a list of the filenames so I can associate them with the DTM
(FileNames <- list.files(TextCorpusPath, pattern=NULL))
NCorpus_DTM <- DocumentTermMatrix(NCorpus,
                                  control = list(
                                    stopwords = TRUE, ## remove normal stopwords
                                    wordLengths=c(3, 10), ## get rid of words of len 2 or smaller or larger than 15
                                    removePunctuation = TRUE,
                                    removeNumbers = TRUE,
                                    tolower=TRUE
                                    #stemming = TRUE,
                                  ))

#inspect(NCorpus_DTM)
## Convert to DF
NCorpusDF <- as.data.frame(as.matrix(NCorpus_DTM))
head(NCorpusDF[1:12,1:10])
NCorpus_MAT <- as.matrix(NCorpus_DTM)
(NCorpus_MAT[1:12,1:10])

## WORDCLOUD ##_---------------------------------------
word.freq <- sort(rowSums(t(NCorpus_MAT)), decreasing = T)
wordcloud(words = names(word.freq), freq = word.freq*2, min.freq = 2,
          random.order = F, max.words=30)

## Place the LABELS onto the DF---------------------------------
## Recall that the filenames must be converted and added as labels
FileNames
(Temp1<-strsplit(FileNames,"_"))
#Temp1[[5]][1]
MyList<-list()

for(i in 1:length(FileNames)){
  MyList[[i]] <- Temp1[[i]][1]
  
}

MyList

## Add MyList to the dataframe
(NCorpusDF$LABEL <- unlist(MyList))
(NumCol<-ncol(NCorpusDF))  ## get the num columns
## Have a look to check for the label - which is at the end
NCorpusDF[1:5, (NumCol-5):NumCol] 

#####################################################
## OK - now we have labeled record data
## and labeled text data
## Both as dataframes..
##
######################################################
NCorpusDF[1:3, ((NumCol/2)-5):(NumCol/2)]   #label here is called LABEL
## AND
head(RecordDF)   ## label here is called "Decision"

##################### CRITICAL ###################
## Make sure that your labels - so LABEL in the text data
## and Decision in the record data are both 
## type FACTOR!!
##################################################
str(NCorpusDF$LABEL)
## It was not!!
## Fix it
NCorpusDF$LABEL <- as.factor(NCorpusDF$LABEL)
## check it
str(NCorpusDF$LABEL)

str(RecordDF$Decision)
RecordDF$Decision <- as.factor(RecordDF$Decision)
str(RecordDF$Decision)
## This one is OK

########################################################
##
##       Decision Trees, etc..
##
#########################################################
##
## At this point, we have created two dataframes - 
## One from record data and one from text data - 
## and both with labels. 
## Now we can perform DT.......................
########################################################
## In R - unlike Python - you only need to remove the label
## from the TESTING data but not from the TRAINING data

## There are many ways to create Training and Testing data
##################### CREATE Train/Test for Record data
##--------------------------------------------------------
(every7_indexes<-seq(1,nrow(RecordDF),7))
##  [1]   1   8  15  22  29  36  43  50  ...
(RecordDF_Test<-RecordDF[every7_indexes, ])
(RecordDF_Train<-RecordDF[-every7_indexes, ])

## There are many ways to create Training and Testing data
##################### CREATE Train/Test for Text data
##--------------------------------------------------------
(every7_indexes<-seq(1,nrow(NCorpusDF),7))
##  [1]   1   8  15  22  29  36  43  50  ...
NCorpusDF_Test<-NCorpusDF[every7_indexes, ]
NCorpusDF_Train<-NCorpusDF[-every7_indexes, ]
NCorpusDF_Train[1:5, (NumCol-5):NumCol] 
NCorpusDF_Test[1:5, (NumCol-5):NumCol] 

#############
## REMOVE the labels from the test data
############

## AFTER
## VERY IMPORTANT - this works when the LABEL is AT THE END...
## Otherwise it must be updated.....
## -------------------------------
(NumCol=ncol(NCorpusDF_Test))
(NCorpusDF_TestLabels<-NCorpusDF_Test[NumCol] ) ##keeping my label
NCorpusDF_Test<-NCorpusDF_Test[-c(NumCol)] ## removing the label from the test set
(NewNumCol<-ncol(NCorpusDF_Test))
NCorpusDF_Test[1:5, (NewNumCol-5):NewNumCol] 

###############--------------------->
## Now remove labels from the Record data test set
###############

(RecordDF_TestLabels<-RecordDF_Test$Decision )
RecordDF_Test<-subset( RecordDF_Test, select = -c(Decision))
head(RecordDF_Test)


##############################################################
##
##            Decision Trees 
##
##############################################################
## Remember the NAMES of the dataframes:
## RecordDF_Train
## RecordDF_Test
## RecordDF_TestLabels
##    *AND*
## NCorpusDF_Train
## NCorpusDF_Test
## NCorpusDF_TestLabels
###################################################################
## Record DT
fitR <- rpart(RecordDF_Train$Decision ~ GPA + Gender + WritingScore, data = RecordDF_Train, method="class")
summary(fitR)

## Text DT
fitText <- rpart(NCorpusDF_Train$LABEL ~ . -night - movie -plot - people, data = NCorpusDF_Train, method="class")
summary(fitText)

##########################
## Predict the Test sets
#########################
## Record--------------------------------
predictedR= predict(fitR,RecordDF_Test, type="class")
## Confusion Matrix
table(predictedR,RecordDF_TestLabels)
## VIS..................
fancyRpartPlot(fitR)

## Text--------------------------------------
(predictedText= predict(fitText, NCorpusDF_Test, type="class"))
## Confusion Matrix
#str(NCorpusDF_TestLabels)
#str(predictedText)
table(unlist(predictedText) ,unlist(NCorpusDF_TestLabels))
## VIS..................
fancyRpartPlot(fitText)


## Save the Decision Tree as a jpg image
jpeg("DecisionTree_Students.jpg")
fancyRpartPlot(fitR)
dev.off()