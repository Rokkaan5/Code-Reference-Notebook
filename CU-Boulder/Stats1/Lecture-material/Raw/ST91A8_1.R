
#*****************************************************************************
#                                 Testing Population Means
#*****************************************************************************

#-----------------------------------------------------------
# One sample Z-testing of population means in R
#------------------------------------------------------------

#One sample tests for (population) means

#In trying to describe the behavior of a population,
#we sometimes need to ask if the mean of the population
# is equal to some hypothesized value

#We use a sample drawn from that population
#to perform the comparison

#There are two cases for one sample tests of significance
#of population variable means


#Case 1: when the population variable standard deviation
#is known and your sample size is large enough: here we use the Z test

#Case 2: when the population standard deviation 
#is unknown and you have a small sample: here we use the t test
#

#----------------------------------------------------------------
#Case 1: Z test for means - population standard deviation 
#is known and sample size is large enough
#----------------------------------------------------------------

#Base R does not have a fucntion for z.test for means so
#we install and load the TeachingDemos package that has that function
install.packages("TeachingDemos")
library("TeachingDemos")

#Example 1: simulating a random sample

#We can create one random sample from a normal 
#population of mean = 3 and standard deviation = 4

#Note that we know the mean and standard deviation

sample_1 <-rnorm(38,mean=3, sd=4) #note that n>30 here

#We can then use this random sample to test if
# the population mean (i.e. the true mean) is equal to
# 1 (the hypothesized population mean value) in the null hypothesis
#Note that this is an example of a two-sided/two-tailed test

#Steps

#1. Write out your null and alternative hypothesis
#2. Choose your alpha (significance level) and confidence interval
#3. Perform your test and obtain the p-value associated with your test statistic
#4. Compare your p-value with your alpha
#5. Write your finding
#6. Write your conclusion in layman's English

#Case 1: If p-value <= alpha, 

#Reject null hypothesis and conclude that there is evidence in favor of
# the alternative hypothesis


#Case 2: If p-value > alpha

#Fail to reject the null hypothesis and conclude there is evidence in favor
#of the null hypothesis


#H0: true mean (population variable mean) = 1
#H1: true mean (population variable mean) <> 1 (<> is "not equal to")

#alpha = 5% (0.05)

z.test(sample_1,mu=1, sd=4, conf.level=0.9)

#We see that the p-value from the test is less than our alpha of 0.05
#so we reject the null hypothesis and say that there is evidence that
#the true mean (i.e. population variable mean) is not equal to 1

#Write your finding below:

# the p-value < alpha. So we reject the null hypothesis

#Write your conclusion below:

#There is evidence that the true mean is not equal to 1

#Example 2: IQ Scores

#Suppose we have a vector (sample of) of IQ scores as follows:

IQ_data <-c(100, 101, 104, 109, 125, 116, 105, 108,110)

#H0: true mean =100
#H1: true mean <>100

#alpha = 5% (0.05)

#Suppose we know that the standard deviation of 
#IQ score variable in the population is 15
# We can test if the population mean is 100 in a null hypothesis
#against the alternative that it is not 100

z.test(IQ_data,100, 15)
z.test(IQ_data, 100, 15, alt="two.sided")#two-tailed version

#Finding/result

#p-value is > alpha (0.05), so we fail to reject the null hypothesis

#Conclusion

# there is evidence that the true/population mean is 100
# There is evidence in favor of the null

# We can conduct a one-sided significance test (i.e. if the population mean is 
# greater than or equal to 100 in a null hypothesis)
#against the alternative that it is less than 100

#H0: true mean (population IQ mean) >=100
#H1: true mean (population IQ mean) < 100

#alpha = 5%

z.test(IQ_data,100, 15, alt="less")# One-tailed version

#----------------------------------------------------------------
#Case 2: t test for means - population standard deviation 
#is not known or sample size is small - or both
#----------------------------------------------------------------
 
#Note that we also use the t test when we have
#small random samples (i.e. n<30)

#The t-distribution is defined by degrees of freedom,
#which are related to sample size
#The t-distribution is most useful for small sample sizes or 
#when the population standard deviation is not known - or both

#As the sample size increases, the t-distribution shape looks more
#like a normal distribution shape

#Similarly, as the degrees of freedom approaches infinity
# the shape of a t-distribution approaches the shape of 
#the normal distribution


#To calculate the degrees of freedom for a question
#use n-1, where n is sample size
#Degrees of freedom relate to the maximum number
#logically independent values (values that have the freedom to vary)
#in a data sample

#We can use the t-distribution (a.k.a Student's t) family
#when we have small samples

#Just as you can use d,p,q and r prefixes for the normal
#distribution family in R, you can also use the following
#for the t-distribution family

#dt() - to get density
#pt() - for cumulative density, cdf
#qt() - for quantiles
#rt() - random number generation

#Note that a variable with t-distribution is distinguised
# from another variable with t-distribution by their
#degrees of freedom


#Note that the t.test() function exists in Base R

#Example 1

#Suppose The United States Department of Agriculture (USDA)
#provides the following sample data for wheat crop yields from across
#various farm plots in the US. Note data is in bushels per plot

yield_data <-c(2.5, 3.0, 3.1, 4.0, 1.2, 5.0, 4.1,3.9, 3.2, 3.3,2.8,4.1,2.7,2.9,3.7,2.6)

# There is a debate within the USDA: leadership believes the mean population 
# yield in the population of farms across the US is 2 bushels per plot
# while the USDA research team believes it is higher than 2

#To settle the debate, the USDA hires you as an external research consultant
# to perform a test of significance for the population mean and test the following
# at a 5% significance level


#H0: Mean population yield is 2
#H1: Mean population yield is greater than 2

#alpha = 5% (0.05)

#Write R code to conduct the test

#Write your answer below:

t.test(yield_data, alternative="greater", mu=2)


#Question:

#What is your finding/result from the test?



#Write your answer below:

#p-value < alpha. So we reject the null hypothesis



#Question:

#What is your conclusion?

#Write your answer below:

#Since the p-value < alpha, there is evidence that the 
#mean population yield is greater than 2


#Example 2

#Generate one random sample with rnorm() but pretend you do not know
#the population standard deviation

sample_2 <-rnorm(15,mean=3, sd=4) #note that n<30 here

#Test, at the 5% significance level
# the null hypothesis that the true mean is less than or equal to 2
#against the alternative that the true mean is greater than 2

#H0: true mean <= 2
#H1: true mean >2

#alpha = 5% (=0.05)

t.test(sample_2,mu=2, conf.level=0.9, alternative="greater")

#Compare p-value of test versus chosen alpha (5% or 0.05)

#Write your finding

#Write your conclusion in layman's terms

#Example 3

#Suppose we have a small vector (one random sample) 
#of height data for 4th graders (population) in the US

#We do not know what the standard deviation of height
#data for 4th grade is in the population
#We are asked to test if the population mean is less than or
#equal to 4.2 ft or greater than 4.2 using the small sample we have
#at alpha level 0.05 and confidence level of 0.95

#H0: mean population height <=4.2
#H1: mean population height > 4.2

#alpha = 5% (0.05)

sample_3 <-c(3.9, 3.1, 4.5, 4.3, 4.0, 3.42, 3.3, 3.4,4.1, 4.3,3.9)

t.test(sample_3, mu=4.2,conf.level=0.95, alternative="greater")

#Finding/result?

#p-value > alpha. So we fail to reject the null hypothesis

#Conclusion?

#Since p-value > alpha and we fail to reject the null hypothesis,
# there is evidence that the true mean is less than or equal to 4.2



#Note that alternative or alt can have the following values
# alt="less", alt="greater", alt= "two.sided" (the default in R)


#H0: mean population height = 4.2
#H1: mean population height <> 4.2

#alpha = 5% (0.05)

sample_3 <-c(3.9, 3.1, 4.5, 4.3, 4.0, 3.42, 3.3, 3.4,4.1, 4.3,3.9)

t.test(sample_3, mu=4.2,conf.level=0.95, alt="two.sided")


#So the p-values under a one-sided test (i.e."greater") suggests we
# fail to reject the null hypothesis whereas under a 
#two sided test suggest we can safely reject the null hypothesis

#How can we tell whether it is a one-tailed or a two-tailed test? 
#It depends on the original claim in the question. 
#A one-tailed test looks for an "increase" or "decrease" 
#in the population parameter 
#whereas a two-tailed test looks for a "change" (could be increase or decrease) 
#in the population parameter.

#Compare p-value of test versus chosen alpha (5% or 0.05)


#Finding/result?

#p-value < alpha. So we fail to reject the null hypothesis

#Conclusion?

#Since p-value < alpha and we reject the null hypothesis
# there is evidence that the true mean is not equal to 4.2

#Example 4: 

#Suppose a consumer group wishes to see whether the 
# actual mileage of a new SUV matches 
# the advertised 17 miles per gallon. The group suspects it is lower
# To test the claim, the group fills the SUV's tank and records 
# the mileage. This is repeated 10 times.
#The results are presented in the variable mpg below:

mpg <- c(11.4,13.1,14.7,14.7,15.0,15.5,15.6,15.9, 16.0, 16.8)

xbar <-mean(mpg) #compute mean
s<-sd(mpg) #compute standard deviation from the mean
n<-length(mpg) # compute number of rows/observations

#To settle the debate, we can conduct a one sample significance test 
#of the null that mean mpg is greater than or equal to 17 against
#alternative that mean mpg is less than 17 at a 5% significance level

#Note that since we have the alternative as less than in the 
#question, we have to perform a one-sided test

#H0: true mean mpg is greater than or less than 17
#H1: true mean mpg is less than 17

# alpha = 5% or 0.05

t.test(mpg, mu=17, alt="less")# one-tailed version

#Compare p-value of test versus chosen alpha (5% or 0.05)

#Write your finding

#Write your conclusion in layman's terms

#Example 5: 

#A college bookstore claims that, on average students
#will pay 500 per semester for textbooks and supplies.
# A student group investigates this claim by randomly
#selecting and interviewing twelve students and finding the amount
# of spent on textbooks and supplies for each. 
#The observations collected are:

#*********************************************************************
#*Costs ($): 304, 431, 385, 987, 303, 480, 455, 724, 642, 506, 507,451
#*********************************************************************

#Question 1

#Do a one sample significance test at 5% of the null that 
# mean costs is equal to $500 versus/against the 
#alternative that mean costs is greater than $500


#Perform the tests above at
#alpha = 0.05
#Confidence level of 95%


#Let cost be the object that holds the data above

#Code for Question 1

cost <-c(304, 431, 385, 987, 303, 480, 455, 724, 642, 506, 507,451)

t.test(cost, mu=500, conf.level=0.95, alt="greater")

#the p-value suggests we fail to reject the null

#Example 6: 

#Suppose the US Department of Energy conducts weekly
#phone surveys on the price of gasoline sold in the US. Suppose
# one week the sample average was $4.03, the sample 
#standard deviation was $0.42 and the sample size is 800. 
#Perform a one-sided significance test of the null that
#the population mean gas price is $4.00 versus/against 
#the alternative that it is greater than $4.00
#Working with the t-distribution in R


#Since we do not know the population standard deviation,we can use 
#the t-test here

#Lets first generate a random sample of size 800 with t-distribution

gas <-rt(800, 799) #sample size 800, with degrees of freedom 799

gas

gas_2 <-rnorm(800, m=4.03, s=0.42) #Use given sample statistics

head(gas_2)

head(gas)

t.test(gas, mu=4.00, conf.level=0.95, alt="greater")

t.test(gas_2, mu=4.00, conf.level=0.95, alt="greater")

#We can use the t-distribution (a.k.a Stuent's t) family
#when we have small samples


