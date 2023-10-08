
#*****************************************************************************
#                          Week: April 17 - 21, 2023
#*****************************************************************************

#######################################
# Plotting a normal curve in R
#######################################


#Load Tidyverse
library(tidyverse)

#load dplyr
library(dplyr)

#load ggplot2
library(ggplot2)

#dnorm() is a useful tool for plotting
# a normal distribution (or a variable with a normal distribution)

#Lets use dnorm() and ggplot() to create and draw a variable (IQ_values) that 
# is thought to have a normal distribution in the real world

#Create a Variable for the x-axis (Call it IQ)

IQ_values <-seq(40,160,1) #This creates a a sequence of 121 numbers from 40 to 160

#take/check for the mean and standard deviation of variable x_values

length(IQ_values) #counts the number of values in x_values
mean(IQ_values) #takes the mean of x_values

#Space IQ_values with the "agreed" standard deviation for IQ (i.e. 15)

IQ_sd <-seq(40,160,15)#creates 9 standard deviation values from 40

#The code creates ggplot for normal probability density function 
#with mean 100 and standard deviation of 15

ggplot(NULL, aes(x=IQ_values, y=dnorm(IQ_values,m=100,s=15)))+
  geom_line()+
  labs(x="IQ", y="f(IQ)")

#Rescale the IQ_values axis to ignore the default values by ggplot and 
#insert the actual values for IQ_values i.e. from 40 with increments of 15

ggplot(NULL, aes(x=IQ_values, y=dnorm(IQ_values,m=100,s=15)))+
  geom_line()+
  labs(x="IQ", y="f(IQ)")+
  scale_x_continuous(breaks=IQ_sd, labels=IQ_sd)#adds the labels we want

#We can also generate a variable that is guaranteed to be normally distributed
#We use the rnorm() function to generate values that are normally distributed
#Syntax for rnorm() is rnorm(n,m,s) where:

#n is number of observations
#m is mean of observations
#s is standard deviation of observations

#Example - generate 19000 random values that are normally distributed.
#with mean 25 and standard deviation 45
#Assign the observations or values to a variable V_norm

V_norm <-rnorm(19000, m=25, sd=45)
V_mean <-mean(V_norm)
V_sd <-sd(V_norm) #Verifies the standard deviation of V_norm

V_mean
V_sd

#Plot V__norm using ggplot: Option 1

ggplot(NULL, aes(x=V_norm, y=dnorm(V_norm,m=25,s=45)))+
  geom_line()+
  labs(x="V_norm", y="f(V_norm")
 

#Plot V__norm using ggplot: Option 2

ggplot(NULL, aes(x=V_norm, y=dnorm(V_norm,m=25.54519,s=44.83144)))+
  geom_line()+
  labs(x="IQ", y="f(IQ)")+
  scale_x_continuous(breaks=V_sd, labels=V_sd)#adds the labels we want
  


##############################################################
#*               Cumulative Density Function
##############################################################
#*
#Cumulative density function

#The cumulative density function (CDF) in R pnorm(x,m,s) returns the
#probability of a score or observation less than x in 
#a variable with a normal distribution with mean m and standard deviation s
#(i.e. a variable with a normal probability density function)

#Remember that the CDF represents the area under the
#probability density function (PDF) for a variable with a normal distribution
#Also remember that the total area under the PDF equals 1 (or 100%)
#So the probability for a score less than a value for the variable
#on the x-axis is given by the following code in R: pnorm(x,m,s)

pnorm(100,m=100,s=15)
#Code above returns the probability of a score less than 100 (i.e. P X<=x) 
#for a normally distributed variable with mean 100 and standard deviation 15

pnorm(85,100,15) #returns probability of getting a value less than 85

#To find the probability of a score/value greater than 85
# use argument lower.tail. It has default value TRUE
# lower.tail = TRUE means "less than"
# lower.tail = FALSE means "greater than"

pnorm(85,m=100,s=15, lower.tail=TRUE)#returns probability of getting a value LOWER than 85

pnorm(85,m=100,s=15, lower.tail=FALSE)#returns probability of getting a value ABOVE 85

#Notice both probabilities add up to 1 (or 100%)

0.1586553+0.8413447

#We can also find the probability of a score between a lower bound
#and an upper bound e.g. probability of a score between 85 and 100

#Method 1 - using two pnorm() functions

pnorm(100, m=100, s=15, lower.tail=TRUE) - pnorm(85, m=100, s=15, lower.tail=TRUE)

#We can illustrate the workings of the two pnorm() functions

#Method 2 - Using tigerstats package

#We first install a package called tigerstats
#Then we use pnormGC()

install.packages("tigerstats")
library("tigerstats")

#Give the probability of a score between 85 and 100 i.e.P(85 < X < 100)
pnormGC(c(85,100), region="between", m=100,s=15)

#Give the probability of a score between 85 and 100 and plot it
pnormGC(c(85,100), region="between", m=100,s=15,graph=TRUE)

#Other Examples using Method 2

#EX 1

#Give the probability of a score between 20 and 50
pnormGC(c(20,50), region="between", m=100,s=15)

#Give the probability of a score between 20 and 50 and plot it
pnormGC(c(20,50), region="between", m=100,s=15,graph=TRUE)

#EX 2

#Give the probability of a score between 20 and 125
pnormGC(c(20,125), region="between", m=100,s=15)

#Give the probability of a score between 20 and 125 and plot it
pnormGC(c(20,125), region="between", m=100,s=15,graph=TRUE)

#EX 3

#Give the probability of a score between 100 and 150
pnormGC(c(100,150), region="between", m=100,s=15)

#Give the probability of a score between 20 and 50 and plot it
pnormGC(c(100,150), region="between", m=100,s=15,graph=TRUE)

#EX 4 - Most people (68%) are said to have an IQ score of between 85 and 115

#Give the probability of a score between 85 and 115
pnormGC(c(85,115), region="between", m=100,s=15)

#Give the probability of a score between 20 and 50 and plot it
pnormGC(c(85,115), region="between", m=100,s=15,graph=TRUE)


#################################################
#* Quantiles of normal distributions
################################################

#The qnorm() function is the inverse of pnorm().
#If you give qnorm() and area, it will return the score
#that cuts off that area (to the left) in the specified
#normal distribution

#The area of the normal curve for IQ scores less than the mean is 0.5

pnorm(100, m=100, sd=15)

#0.5 (or 50%) says that the score 100 is the 50th percentile (median)

pnorm(123, m=100, sd=15)

#0.9374031 says that the score 123 is the 93.37th percentile

pnorm(160, m=100, sd=15)

#0.9999683 says that the score 160 is the 99.999th percentile
#Albert Einstein is said to have an IQ score of 160 
#which is clearly an outlier in the normal distribution of IQ scores

#So if we know the percentile, we can compute the number that sits at
#that percentile using qnorm() or qnormGC() (from the tigerstats package)

#Example

#We know that the probability that a score is less than 85 is 0.1586553
# which means that the 15.86th percentile is 85

pnorm(85, m=100, s=15)

#So 85 is the score that cuts off that area (0.1586553 
# or 15.86% of the total area under that particular normal curve)
# to the left

#Note that you can interpret probability (0.1586553) as percentile (15.86th percentile)
# and also as percentage of total area under particular normal curve (15.86%)

#So for the area 0.1586553 under normal curve with m=100 and sd=15:

qnorm(0.1586553,m=100,sd=15)

#We see that the 15.86553rd percentile is 85

#Lets use the qnormGC (in the tigerstats package) to find that number

qnormGC(0.1586553, region="below", m=100, sd=15)

qnormGC(0.1586553, region="below", m=100, sd=15, graph=TRUE) #graphs it


#For the area 0.5 (or 50% under the normal curve with m=100 and sd=15:


qnormGC(0.5, region="below", m=100, sd=15)

qnormGC(0.5, region="below", m=100, sd=15, graph=TRUE) #graphs it

#So the 50th percentile is 100 - meaning 
# 50% of all the scores for the variable with mean 100 and standard deviation 15
# are under 100

#For the area 0.9999683 under normal curve with m=100 and sd=15:

qnormGC(0.9999683, region="below", m=100, sd=15)

qnormGC(0.9999683, region="below", m=100, sd=15, graph=TRUE) #graphs it

#So the 99.9999683rd percentile is 159.9 or 160 - meaning 
# 99.9999683rd of all the scores for the variable with mean 100 and standard deviation 15
# are under 159.9 - which is why we think Albert Einstein is a genius

# Statisticians are sometimes concerned with the 25th percentile (1st quartile)
# the 50th percentile (2nd quartile), 75th percentile (3rd quartile), and
# 100th percentile (4th quartile)

#To find the 1st, 2nd, 3rd and 4th quartiles for a normally distributed variable with
# mean 100 and standard deviation 15:

qnorm(c(0.25,0.5,0.75,1.00), m=100, sd=15)

round(qnorm(c(0.25,0.5,0.75,1.00), m=100, sd=15))

#To find the 0th percentile, 25th percentile,50th percentile,75th percentile, and
#100 percentile for a normally distributed variable with
# mean 100 and standard deviation 15:

qnorm(c(0, 0.25,0.5,0.75,1.00), m=100, sd=15)

round(qnorm(c(0, 0.25,0.5,0.75,1.00), m=100, sd=15))

#Note that the 0th percentile is minus infinity, and the
#100th percentile is plus infinity
#These numbers (-inf and +inf) suggest that
#the CDF never completely touches the x-axis nor reaches a maximum. 
#Thus the tails of the normal curve are said to be asymptotic to the x-axis

###########################################
#The standard normal distribution in R
###########################################

#A normally distributed variable can be 
# converted into a variable with a standard normal
#distribution by generating the Z-scores for each
#observation

#The standard normal distribution in R is a 
#member of the normal distribution family
#where the mean is 0 and the standard deviation is 1

#The Z-score numbers make the variable a 
# variable with a standard normal distribution
# which means that its mean is 0 and its 
# standard deviation is 1


#Working with variables with standard normal distributions
# is easy in R: when we use the norm() family functions
#we do not specify a mean and standard deviation
# the defaults are 0 and 1

#Examples

#Find the density value for the value 3 for normal variable 
# with mean 4 and standard deviation 2.5

pnorm(3, m=4,s=2.5) #base R

install.packages("tigerstats")
library("tigerstats")
pnormGC(3,m=4,s=2.5) #Using pnormGC in tigerstats package

#Find density value for the value 2 for a variable with standard
#normal distribution

pnorm(2) #If you do not specify mean and sd, R assumes standard normal

pnorm(2,m=0,s=1) #You can choose to specify mean and sd


#Find the 25th, 50th and 75th percentiles for a
#normal variable with mean 4.2 and standard deviation 1.1

qnorm(c(.25,.50,.75), m=4.2, s=1.1)


#Find the 25th, 50th and 75th percentiles for a
#standard normal variable 


qnorm(c(.25,.50,.75))

#Notice the median (=mean and mode in a standard normal) is 0

#Generate 7 observations for a normally distributed variable
#ensure the mean is approximately 3 and standard deviation is 2.3
#assign results to variable normal_variable

normal_variable <-rnorm(7, m=3,s=2.3)

mean(normal_variable)
sd(normal_variable)


#Generate 7 observations for a standard normal variable
#Check the mean and standard deviation
#assign results to variable st_normal_variable

st_normal_variable <-rnorm(7)

mean(st_normal_variable)
sd(st_normal_variable)


# In the real world, when we deal with small samples
# the normal distribution is not very appropriate
# You pay a price for using a small sample:
# you have a large standard error

# For small samples, the sampling distribution
# of the means is a member of the t-distribution
#family.

#Unlike the normal distribution family distinguished by
#mean and standard deviation, members of 
#the t-distribution family are distinguished by
#degrees of freedom

# Like in the case for the normal family - dnorm(),
# pnorm(), qnorm(), rnorm(), we also have the functions for
# for the t-distribution family

#dt() - density function
#pt() - cumulative density function
#qt() - quantile
#rt() - random number generation with t-distribution



#*****************************************************************************
#                                   Sampling Using R
#*****************************************************************************

###############################################
#Simple Random Sampling
##############################################


#Case 1: Where the population has a sequence of numbers
#-----------------------------------------------------------

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


#Case 2: case were population does not follow a sequence
#--------------------------------------------------------

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

#Systematic sampling

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

#Example 1
#Systematic sampling in R

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



#Example 2
#Systematic sampling in R

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


# According to the CLT
# 1. The sampling distribution of the mean
# is approximately normal if the size of the ramdom
# sample is large enough
# (large enough means about 30 or more)

# 2. The mean of the sampling distribution
# of the mean is the same as the population mean

# 3. The standard deviation of the sampling distribution
# of the mean (also known as the standard error
# of the mean) is equal to the population standard deviation
# divided by the square root of the sample size

# 4. If the population that supplied the sample
# is not normal, the sampling distribution of the mean
# will be a normally distributed variable so long as the sample size
# is large enough - for practical purposes, 30 or more

# 5. If the population that supplied the sample
# is normal, then the sampling distribution 
# will be a normally distributed variable for any sample size

#Note that the population that supplied
#The samples does not have to be normally
#distributed for the CLT to hold. The key
#requirement is that the size of the 
#samples drawn from the population
#is large enough


#The CLT specifies (approximately) that large
# samples will be normally distributed

# In the real world, when we deal with small samples
# the normal distribution is not very appropriate
# You pay a price for using a small sample:
# you have a large standard error

# For small samples, the sampling distribution
# of the means is a member of the t-distribution
#family.

#Unlike the normal distribution family distinguished by
#mean and standard deviation, members of 
#the t-distribution family are distinguished by
#degrees of freedom

# Like in the case for the normal family - dnorm(),
# pnorm(), qnorm(), rnorm(), we also have the functions for
# for the t-distribution family

#dt() - density function
#pt() - cumulative density function
#qt() - quantile
#rt() - random number generation with t-distribution

