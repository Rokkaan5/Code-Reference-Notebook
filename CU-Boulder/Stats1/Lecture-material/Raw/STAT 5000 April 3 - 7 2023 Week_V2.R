
#*****************************************************************************
#                       STAT 5000 Week: April 3 - 7 2023
#*****************************************************************************

#Install NYC Flights Package
#Install nycflights from Tools>Install Packages> nycflights


#nycflights13 contains information about domestic flights departing from New York City (NYC)
# 3 main airports in 2013: Newark Liberty International (EWR), John F. Kennedy International (JFK)
#and LaGuardia Airport(LGA)

#nycflights13 is an R Package contains five datasets saved in five dataframes as follows:

#1. flights: Information on all 336,776 flights
#2. airlines: A table matching airline names and their two-letter International Air Transport Association (IATA) 
#   airline codes (also knon as carrier codes) for 16 airline companies. For example "DL" is the two-letter code for  Delta
#3. planes: Information about each of the 3,322 physical aircract used
#4. weather: Hourly meteorological data for each of the three NYC airports
#5. airports: Names, codes, and locations of the 1,458 domestic destinations

#load Tidyverse and nycflights13 packages

install.packages("nycflights13")
library(tidyverse)
library(nycflights13)

#Load Tidyverse
library(tidyverse)

#load dplyr
library(dplyr)

#load ggplot2
library(ggplot2)

#Lets work with the flights Data Frame (or Tibble - a better, well-behaved data frame in Tidyverse )

#call up the Tibble flights

flights

class(flights)

#Use dim to check rows and columns; use glimpse to get
#additional data about data types





dim(flights)

glimpse(flights)

#Use view to bring up RStudio built-in data viewer

view(flights)



# We will use 5 key charts or plots to explore the contents of the flights data frame/tibble

# 1. Scatterplots
# 2. Linegraphs
# 3. Histograms
# 4. Boxplots
# 5. Barplots

# In using ggplot, we can break a graphic into 3 components

#1. data: the dataset containing the variables of interest
#2. geom: the geometric object (e.g scatterplot) in question
#3. aes: aesthetic attributes of the geometric object

# Chart 1. Scatterplots - allow you visualize the relationship between 2 variables
#Use ggplot() + geom_point()

# lets look at plotting 2 variables for Alaska Airlines only:
# dep_delay: departure delay on the horizontal "x" axis 
# arr_delay: arrival delay on the vertical "y" axis

#Filter out Alaska Airlines data from flights data frame
alaska_flights <-flights %>% filter(carrier=="AS")

ggplot(data = alaska_flights, mapping=aes(x=dep_delay, y=arr_delay))+
  geom_point()

#Addressing Overplotting (1) increase transparency (2) Jittering the points

#Option 1: Increasing transparency
ggplot(data = alaska_flights, mapping=aes(x=dep_delay, y=arr_delay))+
  geom_point(alpha=0.2)

#Option 2: Jittering the points
ggplot(data = alaska_flights, mapping=aes(x=dep_delay, y=arr_delay))+
  geom_jitter(width = 30, height = 30)

# Chart 2. Linegraphs - Show the relationship between two numerical variables when the
# variabe on the x-axis (also called the explanatory variable), is of a sequential nature
# i.e there is ordering to the variable e.g. time (month, year etc)

#Use ggplot() + geom_line()

# lets look at the weather data frame:

dim(weather)
glimpse(weather)
view(weather)

#Filter out data where origin is "EWR", month is January and day is from 1 to 15
#into a data frame/Tibble called early_january_weather 

early_january_weather <-weather %>%
  filter(origin=="EWR" & month==1 & day <=15)

#Check the data you just extracted

early_january_weather

head(early_january_weather)
tail(early_january_weather)

dim(early_january_weather)
glimpse(early_january_weather)
view(early_january_weather)

#Plot the linegraph of temperature over time

ggplot(data=early_january_weather, mapping=aes(x=time_hour, y=temp))+
  geom_line()

ggplot(data=early_january_weather, mapping=aes(y=temp, x=time_hour))+
  geom_line()

#what can you conclude about this graph?

# What is happening as the time increases on the x-axis? 
# What is the apparent trend of temperature?


# Chart 3. Histograms - Help us show the distribution of data i.e.
# 1. What are the smallest and largest values?
# 2. What is the "center" or "most typical value"?
# 3. How do the values spread out?
# 4. What are frequent and infrequent values?

#Suppose we want to see the distribution of the variable "temp"

#Use ggplot() + geom_histogram()

ggplot(data=weather, mapping=aes(x=temp))+
  geom_histogram()

# First warning message tells us that the histogram was constructed using a default value of 30
# Second message tells us that because one row has a missing value (NA) for temp,it was removed
# before generating the histogram

#To get a better sense of the range of temperatures, we can add white vertical borders to demarcate the bins

ggplot(data=weather, mapping=aes(x=temp))+
  geom_histogram( color="white")

# You can change the bin colors to "blue steel"

ggplot(data=weather, mapping=aes(x=temp))+
  geom_histogram(color = "white", fill="steelblue")

#Adjusting the bins (1) adjusting number of bins with bins argument to geom_histogram 
# (2) adjusting the width of the bins via the binwidth argument to geom_histogram

#(1) Adjustment using the bin argument

ggplot(data=weather, mapping=aes(x=temp))+
  geom_histogram(bins=40, color = "white", fill="steelblue")

#(2) Adjusting using the binwidth argument

ggplot(data=weather, mapping=aes(x=temp))+
  geom_histogram(binwidth = 10, color = "white", fill="steelblue")



# NOTE: Histograms, unlike Scatterplots and Linegraphs present information on only 
# a single numerical variable, specifically the visualization of the distribution of
# that particular variable

#Faceting histograms by another variable
#Suppose we want to see the distribution of the variable "temp" across months in a year
#The variable "month" exists in the data frame and can serve as a variable for this

glimpse(weather)


ggplot(data=weather, mapping=aes(x=temp))+
  geom_histogram(binwidth=5, color = "white", fill="steelblue")+
  facet_wrap(~month)


#Which of the distributions is closest to the popular normal distribution?

#NOTE: Faceted histograms help us compare the distribution of a numerical variable
#split by the values of another variable


# Chart 4. Boxplots - Like Histograms help us show the distribution of data i.e.
# 1. What are the smallest and largest values?
# 2. What is the "center" or "most typical value"?
# 3. How do the values spread out?
# 4. What values are outliers in the data?

#Boxplots visually summarise data by cutting observations into quartiles 
#(4 buckets): 
# A Boxplot is constructed from the 5-number summary of a numerical variable

ggplot(data=weather, mapping=aes(x=factor(month), y= temp))+
  geom_boxplot()

# The "box" portions of the visualization represent the 1st quartile,
#the median (2nd quartile), and the 3rd quartile

#NOTE the following:
  
# 1. The height of each box is the value of the 3rd quartile minus the 1st quartile
# This is called the Interquartile range (IQR)
# 2. The "whisker" portions of these boxplots represent points less than the 1st quartile
# and greater than the 3rd quartile
# 3. The dots represent outliers - values falling outside the whiskers
# they can be thought of as "out-of-the-ordinary" values

# Discussion Questions:

# Which month appears to have the largest spread of temperatures within that month
# What is the trend of median temperature between month 1 to month 12?
# Which month has the largest IQR?
# What is the trend of IQR across months 1 to 12?
# Which months have the largest number of outliers?

# Measures of spread include variance, standard deviation, IQR
# Measures of central tendency include mean, median, and mode


# Chart 5. Barplots - While Histograms and Boxplots are used to visualize
# the distribution of numerical variables, Barplots are used to visualize
# the distribution of categorical variables
#Barplots are also called Barcharts

#NOTE:

# 1. The bars in a Barplot can represent "counts" of cases
# 2. The bars in a Barplot can represent "values" 

# Case 1: Where the Barplot represents value: Use ggplot()+ geom_bar()

glimpse(weather)

ggplot(data=weather, mapping=aes(x=temp))+
  geom_bar() #Shows count of temperatures recorded across all 3 NYC airports in 2013

glimpse(flights)

ggplot(data=flights, mapping=aes(x=carrier))+
  geom_bar() #Shows the number of flights departing all 3 NYC airports in 2013 by carrier/airline

ggplot(data=flights, mapping=aes(x=origin))+
  geom_bar() #Shows number of flights departing all 3 NYC airports (without identifying carriers)

# We can create a stacked Barplot by using the fill argument inside the aes function

ggplot(data=flights, mapping=aes(x=carrier, fill=origin))+
  geom_bar() #Shows number of fligths by carrier/airline and by airport

# Case 2: Where the Barplot represents value: Use ggplot()+ geom_col()

ggplot(data=weather, mapping=aes(x=factor(origin), y=temp))+
  geom_col()

ggplot(data=weather, mapping=aes(x=factor(origin), y=humid))+
  geom_col()

ggplot(data=weather, mapping=aes(x=factor(origin), y=wind_speed))+
  geom_col()

ggplot(data=weather, mapping=aes(x=factor(origin), y=visib))+
  geom_col()

#We can also facet Barplots

ggplot(data=flights, mapping=aes(x=carrier))+
  geom_bar()+
  facet_wrap (~origin, ncol=1) # Facet by carrier and origin in one column

ggplot(data=flights, mapping=aes(x=carrier))+
  geom_bar()+
  facet_wrap (~origin, ncol=2) # Facet by carrier and origin in two columns

ggplot(data=flights, mapping=aes(x=carrier))+
  geom_bar()+
  facet_wrap (~origin, ncol=3) # Facet by carrier and origin in three columns

ggplot(data=flights, mapping=aes(x=carrier))+
  geom_bar()+
  facet_grid (~origin) #Facet by carrier and origin side by side


