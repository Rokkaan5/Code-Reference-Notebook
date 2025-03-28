
#*****************************************************************************
#         Week: April 24 - April 28: Analysis of Variance (ANOVA)
#*****************************************************************************


#Load Tidyverse
library(tidyverse)

#load dplyr
library(dplyr)

#load ggplot2
library(ggplot2)


# Today we discuss hypothesis testing about samples from
# about more than two populations

#This analysis is called Analysis of Variance (ANOVA)

#Analysis of variance (ANOVA) is a method of comparing
# means (behavior) across several populations using several
#samples based on variations from the mean

#If variations from the mean are different for the populations
# it is unlikely that the population means are the same

#We use more than two samples when we have more than two populations

#We will discuss the following cases:

#Case 1. testing hypothesis when samples are independent of each other
#Case 2. testing hypothesis when samples are dependent on each other

#-----------------------------------------------------------------------
#Case 1. testing hypothesis when samples are independent of each other
#-----------------------------------------------------------------------

#***********************************************************************
#Example 1:

#Suppose we have the following weight samples from three different populations
#The samples drawn from these populations are likely independent 
#since they are drawn from three different populations

smpl1 <-c(95,91,89,90,99,88,96,98,95)

smpl2 <-c(83,89,85,89,81,89,90,82,84,80)

smpl3 <-c(68,75,79,74,75,81,73,77)

#Suppose we want to check if the population means are likely the same
# or whether they are they different?

#We can test the following null hypothesis at 5% significance level

#H0: population mean 1 = population mean 2 = population mean 3
#H1: Not H0

#Set alpha = 5% (0.05)

#The procedure for testing more than two populations
#using more than two samples is called Analysis of Variance (ANOVA)

#The R function for ANOVA is aov()

# The syntax for aov() is as follows:
# aov(Dependent_variable ~ Independent_variable, data)


#Suppose we have drawn three samples of weights from three populations:

smpl1 <-c(95,91,89,90,99,88,96,98,95)

smpl2 <-c(83,89,85,89,81,89,90,82,84,80)

smpl3 <-c(68,75,79,74,75,81,73,77)


#Our goal is to answer the question: are weights on average the same
# or different across the three populations from where we obtained the
# samples if we assume equal variances across the three populations?

#We have to convert the data into a data frame to use aov()


#Step 1: create a single vector that consists of smpl1,smpl2,smpl3

single_vector <-c(smpl1,smpl2,smpl3)

single_vector #Note all numbers are now in one variable single_vector

#Step 2: create a vector consisting of the sample names
# matches up against the respective numbers
# i.e. a vector consisting of "smpl1" repeated 9 times,
# "smpl2" repeated 10 times and "smpl3" repeated 8 times


long_vector <-rep(c("smpl1", "smpl2", "smpl3"),
                  times=c(length(smpl1),length(smpl2), length(smpl3)))

long_vector #entries created to help identify the numbers in single_vector


#Step 3: create data frame:

sample_df <-data.frame(long_vector,single_vector)


#Step 4: view the data frame, sample_df

sample_df

#Note that the sample now has the structure of a data frame

#Step 5: conduct analysis of variance 

anova_analysis <-aov(single_vector ~ long_vector, data=sample_df)

#Step 6: Use summary function to display ANOVA results

summary(anova_analysis)

#Write your finding/result

#Compare p-value with your chosen alpha

#We see that the p-value is <= alpha (5%) - so we reject the null (H0)

#Write your conclusion

#We conclude that there is evidence in favor of the alternative hypothesis
# i.e. Not H0
#i.e the means of the populations (from where the samples were drawn)
# are not the same
#i.e. the average behavior of the populations are not the same


#The R function oneway.test() also lets us achieve 
#the same results from aov() above

#The test above is also called one way ANOVA when we assume the 
# variance from the means for all populations are equal

oneway.test(single_vector ~ long_vector, data=sample_df, var.equal=TRUE)

#The advantage of the oneway.test() function is that you write fewer code 
#(e.g you do not need a summary statement to display your ANOVA summary)


#***************************************************************************
#Example 2: Energy drink consumption levels and energy activity levels

#Imagine we wanted to test the effects of energy drinks on students in a university population
#So we ask a sample of students to consume energy drinks in various quantities
#We obtain the following data that captures consumption category and energy level after 
#consumption

#Our goal is to answer the question: are energy levels the same across
# energy drink consumption levels if we assume equal variances for
# the populations?


energy_level <-c(3,2,1,1,4,5,2,4,2,3,7,4,5,3,6)
consumption_level<-gl(3,5, labels = c("No consumption", "Low consumption", "High consumption"))

study_data <-data.frame(consumption_level,energy_level)

#The gl() function is used to generate factor levels - essentially
#create categorical/level values for the categorical variable

#One syntax of the gl() function:

#gl(n, k, length = n*k, labels = seq_len(n), ordered = FALSE)

#Where:

#n: This is an integer that indicates the number of levels
#k: This is an integer that indicates the number of replications
#length: This is an integer that indicates the length of the result
#labels: This is an optional vector of labels for the output 
#of the factor levels
#ordered: This takes a logical value (either TRUE or FALSE), 
#which indicates whether the output should be ordered or not.


#Display data frame

study_data


# R Code to test the claim that energy levels are the same for students in the population
# regardless of energy drink consumption level 

#H0: no change in energy level regardless of energy drink consumption level

#H1: Not H0

#alpha =5% (0.05)

anova2 <-aov(energy_level ~ consumption_level, data=study_data)

summary(anova2)

#Alternative R code is to run the following:

oneway.test(energy_level ~ consumption_level, data=study_data, var.equal=TRUE)


#Note that the assumptions for one-way ANOVA include the following:

#1. Variances are are equal (or homogeneous)
#2. Samples/observations are independent
#3. dependent variable should be measured on at least an interval scale




#------------------------------------------------------------------------
#Case 2: ANOVA for dependent samples - i.e. more than two matched samples
#------------------------------------------------------------------------

#This type of ANOVA is also called repeated measures or 
# randomized blocks or within-subjects


#**********************************************************************
#Example 1

#Repeated measures ANOVA in R

#Suppose we have the following people sign up for a weight-loss program

person <-c("Al", "Bill","Charlie","Dan", "Ed", "Fred","Gary", "Harry", "Irv", "Jon")

#Suppose their respective weight measurements before entering the program are:

before <-c(198,201,210,185,204,156,167,197,220,186)

#Suppose their respective weight measurements after 1 month in the program are:

one_month <-c(194,203,200,183,200,153,166,197,215,184)

#Suppose their respective weight measurement after 2 months in the program are:

two_month <-c(191,200,192,180,195,150,167,195,209,179)

#Suppose their respective weight measurement after 3 months in the program are:

three_month <-c(188,196,188,178,191,145,166,192,205,175)

#Create a data frame to hold the data

initweight_df <-data.frame(person, before, one_month, two_month, three_month)

#View the final data frame containing weight measures for the participants:

initweight_df

#Note that the data is in wide format
#We need to reshape the data into long format
#We use the melt() function to convert the data frame from
# wide format to long format
#Make sure to install and load the "reshape" package

install.packages("reshape")
library("reshape")

finalweight_df <-melt(initweight_df, id="person")

#Label the new columns appropriately

colnames(finalweight_df) = c("person", "time","weight")

#Display the new data frame

finalweight_df

#Conduct a repeated measures ANOVA at 5% significance level

#We can use ANOVA to check the following 
#two cases based on the data frame finalweight_df

# Case A. Check if weight across time 
# is the same or not in the population regardless of person

# Case B. Check if weight across time is the same or not in the population
# per person/accounting for person

#**************************************************************************
# Case A. aov(weight ~ time, data = finalweight_df)

# H0: weight before and across time (i.e. one_month, etc) 
# are equal/ no change from before to other periods

# H1: Not H0

# alpha =5% (0.05)

anova1 <-aov(weight ~ time, data=finalweight_df)

anova1

summary(anova1)


#Finding/result

# p-value (0.634) > alpha (0.05). 
#We fail to reject the null hypothesis H0

#Conclusion

# There is evidence that weight before and across 
# time (i.e. one_month, etc) are equal/ no change

#***********************************************************************
#Case B. aov(weight ~ time + Error(person/time),data = finalweight_df)

#H0: weight before study and across time (i.e. one_month, etc) after study 
# are equal/no change from before to other periods per person

#H1: Not H0

#alpha =5% (0.05)

anova2 <-aov(weight ~ time +Error (person/time), data=finalweight_df)

#The code above allows us do a "within-subjects" ANOVA analysis

#Display ANOVA results

summary(anova2)

summary(anova1)


#Finding/result

# p-value (7.3e-08) < alpha (0.05). 
#We reject the null hypothesis H0

#Conclusion

# There is evidence in favor of the alternative
# i.e. there is evidence that weight before and across 
# time (i.e. one_month, etc) are not equal per person across time



