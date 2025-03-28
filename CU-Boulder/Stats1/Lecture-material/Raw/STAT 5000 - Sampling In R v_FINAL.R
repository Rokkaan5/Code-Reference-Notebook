
#*****************************************************************************
#                                   Sampling Using R
#*****************************************************************************


#Base R offers the standard function sample() to take a sample from datasets

#We focus on the following cases  here where we 
#use sample() from R and other functions from other packages:

#1. Simple random sampling without replacement
#2. Simple random sampling with replacement
#3. Systematic sampling


#The syntax of the sample() function in R for simple random sampling without
#replacement is below:

#-----------------------------------------------------------
#sample(x, size, replace = FALSE, prob = NULL)
#-----------------------------------------------------------

#The Syntax of the sample() function in R for simple random sampling with
#replacement is as below:

#sample(x, size, replace = TRUE, prob = NULL)


#Where:

#x - vector or a dataset
#size - sample size
#replace - TRUE or FALSE
#prob - probability weights

#Ex 1. Example of sampling without replacement with a synthetic dataset
# that is a range between 1 to 10

#create sample between 1:10
data_sample <-sample(1:10)

#print the sample
data_sample

#Ex 2. sample range is 1:10 and samples size is 3
  

data_sample <-sample(1:10, 3)

#print the sample
data_sample

#Note that the sample size must always be less 
#than the range of values (e.g. in this example, 3 <=10)


#Ex 3. Specifying replace = FALSE or replace = F is used to explicitly sample
#without replacement

set.seed(1) #Use this to reproduce the same sequence every time you run the code
data_sample <-sample(1:5,3)

data_sample

set.seed(1)
data_sample <-sample(1:5, 3, replace = F)

data_sample

#--------------------------------------------------------------------------
#The syntax of the sample() function in R for simple random sampling with
#replacement is below:

#sample(x, size, replace = TRUE, prob = NULL)
#--------------------------------------------------------------------------



#Ex 1. 

set.seed(5)
#take random samples with replacement
data_sample <-sample(1:5, 4, replace = TRUE)

data_sample

#Ex 2. 
set.seed(12)
sample(1:12, 5, replace = T) #This example displays the results directly on-screen


#Ex 3. 

set.seed(12)

data_sample <-sample(1:12, 5, replace = TRUE) #Stores result in an object

data_sample #displays the results


#Taking samples from a dataset

#We take a sample from the popular Iris dataset

#set.seed function

set.seed(10)

#take a sample of 10 rows from the Iris dataset (which exists inside R)

data_sample <-sample(1:nrow(iris), 10)

#display the 10 rows you just sampled

iris[data_sample,]


#----------------------------------------------------------
#Reading CSV files from the web and taking row samples
#----------------------------------------------------------

#Install the key packages
install.packages("readxl") #make sure to install this
install.packages("tidyverse")
install.packages("readr") #make sure to install this

#Load the key packages
library(readxl) #make sure to load this after installing it
library(tidyverse)
library(readr)#make sure to load this
library(dplyr)

web_data <-read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/AER/CASchools.csv")
View(web_data) #view your dataset
dim(web_data)#view the dimensions (rows and columns) of your dataset

summary(web_data) #Summarize all the numeric variables in your dataset
summary(web_data$expenditure)#summarize only the variable web_data$expenditure

#The code below takes a sample of rows of web_data

set.seed(100)

#take a sample of 35 rows from the web_data dataset

data_sample <-sample(1:nrow(web_data), 35)

#display the 35 rows you just sampled:

web_data[data_sample,]

#---------------------------------------------------
#Taking samples by setting the probabilities
#---------------------------------------------------

#Let's use this example to illustrate setting probabilities

#Suppose a company is able to manufacture 10 watches. Among these 10 watches
#20% are found to be defective. We wshow this with the code below:

#create a sample with probability of 80% good watches and 20% effective watches

set.seed(23)
sample(c('good','defective'), size=10, replace=TRUE, prob=c(.80,.20))




#create a sample with probability of 80% good watches and 20% 
#effective watches, put the result in an object data_sample and
#display data_sample

set.seed(23)

data_sample <-sample(c('good','defective'), size=10, replace=TRUE, prob=c(.80,.20))

#display results in data_sample
data_sample



###############################################
#Simple Random Sampling
##############################################

#----------------------------------------------------------------------
#Simple random sampling case where the population is a sequence of numbers
#---------------------------------------------------------------------

#Example 1: 

#Given a population of size N=5, list all of the possible 
#samples of size n=3 with R

#Solution - use combn() function

combn(x=1:5, m=3)

#1:5 suggests that the values in the population of interest
# are sequential starting with the number 1


#Given a population of size N=7, list all of the possible 
#samples of size n=4 with R

#Solution - use combn() function

combn(x=1:7, m=4)


#Note that for the syntax you specify m not n

#-----------------------------------------------------------------------
#Simple random sampling case were population does not follow a sequence
#-----------------------------------------------------------------------

#Example 1

#Given a population of size N=5, where X1 = 2, 
# X2=5, X3 = 8, X4 = 12, X5=13, use R to list 
#all of the possible samples of size n=3.

install.packages("PASWR2")
library("PASWR2")

srs(c(2,5,8,12,13), 3)

#OR

popvalues <-c(2,5,8,12,13)
possible_samples<- srs(popvalues,3)
t(possible_samples)  #Transpose the values


#Example 2

#Given a population of size N=7, where X1 = 3, 
# X2=-3, X3 = -8, X4 = 17, X5=-13,X6= 23, X7=109 use R to list 
#all of the possible samples of size n=4.

install.packages("PASWR2")
library("PASWR2")

srs(c(3,-3,-8,17,13,23,109), 4)

#------------------------------------------------------------
#Systematic sampling
#-----------------------------------------------------------

#Recall that systematic sampling is done when
# a statistician has the population of N units/members
# and desires to select every kth value of that population

#To obtain a systematic sample, 

#Step 1:
#choose a sample size n, calculate N/n and let k be the closest
# integer to N/n

#Step 2:
#Find a random number between 1 and k to be the starting point
# for sampling

#Step 3
#Build out the sample elements

#Example 1: Systematic sampling in R

#Suppose a researcher wants a systematic sample of size 10
#from a list / population of 1000 members

#N=1000
# n=10
# k = N/n: 1000/10 = 100

#K=100 implies every 100th member in the population
#will be chosen.

#To pick the first member, select any number between 1 and 100

#We can task R to select the first member and also
# generate the systematic sample list as follows:

seq(sample(1:100,1),1000,100)



#Example 2: Systematic sampling in R

#Suppose a student wants a systematic sample of size 20
#from a list / population of 1000 members

#N=1000
# n=20
# k = N/n: 1000/20 = 50

#K=50 implies every 50th member in the population
#will be chosen.
#We can task R to select the first member and also
# generate the systematic sample list as follows:

seq(sample(1:50,1),1000,50)


set.seed(2) #Used to produce same/reproducible results
FULL <-seq(sample(1:25,1),1000,25)

#We can then find the mean and standard deviation
#of FULL
mean(FULL)
sd(FULL)
