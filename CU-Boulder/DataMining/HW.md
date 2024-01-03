---
layout: page
title: "DataMining Assignments"
permalink: /CUB/DataMining/HW
---

# [CU Boulder](../../CUB.md): [Data Mining](DataMining.md) - Assignment Submissions
(Spring 2023) Data Mining with Professor Di Wu at CU Boulder

These are all my homework and exam submissions. All work correlates (more or less) with each section in the [Lecture Tutorials](Lectures.md) for the class.

## HW 1 
(Associated with [Data Collection/Integration](Lectures.md#01-data-integration) lecture section)

- [Part A - Reasoning](Assignments/HW1/PartA-Reasoning.html)
    - What do you understand by the term "Data Mining"?
    - {Example scenario with CU dining service business model} 
        - What would you do as a Data Miner? Explain the methods of Data collection in this case.
    - List out the challenges or difficulties in Data Collection. How would you mine huge data?
- [Part B - Application](Assignments/HW1/PartB-Application.html)
    - Data collection "tutorial" for job search in STEM
    - Webscraping application of news data for sentimental analysis
        - (I never finished this part (yet))
- [Part C - Reading](Assignments/HW1/PartC-Reading.html)
    - **Paper title:** *Security and Privacy Issues of Big Data*


## HW 2
(Associated with [Data Manipulation](Lectures.md#02-data-manipulation) & [Data Understanding](Lectures.md#03-data-understanding) lecture section)

- [Part A - Reasoning](Assignments/HW2/PartA-Reasoning.html)
    - What is the step "Data Understanding" in statistics, in the Data Mining Pipeline? Why do we need it?
    - What are the measures of central tendency (Ms)? Explain the relevance of each of them with suitable examples.
    - What are the measures of dispersion? Explain the relevance of each of them with suitable examples.
    - What are the measures of similarity? Explain the relevance of each of them with suitable examples.
    - (Optional question) {given an example scenario and dataset} What statistical test would you perform (t-test or z-test) to check the problem statement?
        - (I never answered)
- [Part B - Application](Assignments/HW2/PartB-Application.html)
    - Job-seeking dataset
        - Apply the Python `pandas` package to the data and demonstrate how we can use it to get a better understanding of the data.
    - News data
        - Perform text processing. Label the data as "positive"/"negative" sentiments
            - (I never finished this part)
- [Part C - Reading](Assignments/HW2/PartC-Reading.html)
    - **Paper title:** *Exploratory Data Analysis*

## HW 3
(Associated with [Data Visualization](Lectures.md#04-data-visualization) lecture section)

- [Part A - Reasoning](Assignments/HW3/PartA-Reasoning.html)
    - What do you mean by a normal distribution of data? What happens when the data is not normally distributed? Explain with an example.
    - How is a quantile-quantile plot different from a quantile plot? Illustrate with an example.
    - Identify plot types 
- [Part B - Application](Assignments/HW3/PartB-Application.html)
    - Data visualization application to `fatalities` dataset
    - Data visualization application to our own dataset from HW1
- [Part C - Reading](Assignments/HW3/PartC-Reading.html)
    - **Paper title:** *Perceived usefulness of online customer reviews: A review mining approach using machine learning & exploratory data analysis*

## HW 4

(Associated with [Data Preprocessing](Lectures.md#04-data-visualization) lecture section)

- [Part A - Reasoning](Assignments/HW4/PartA-Reasoning.html)
    - Practice of normalization
        - min-max scaling
        - z-score normalization
        - decimal scaling
    - OLTP or OLAP of example scenarios
    - Peer review data visualization of other colleagues
- [Part B - Application](Assignments/HW4/PartB-Application.html)
    - Data preprocessing application to `fatalities` dataset
        - normalizing
        - discretize continuous features
        - fill missing values
    - Data preprocessing application to own dataset from HW1
- [Part C - Reading](Assignments/HW4/PartC-Reading.html)
    - **Chosen Paper title:** *Data Preprocessing for Supervised Learning* 

## [Exam 1](Assignments/JasmineKobayashiExam1/Exam1.html)
(The [Data Preprocessing Case Study](Lectures.md#data-preprocessing-case-study---boulder-climate) served as the main form of review for this exam)

All questions had a "*Reasoning*" portion that asked interview-like questions that could be common from a job-interview live-coding test. (Ex: all of them had "Is there any difficulty you met in finishing this question? If yes, how did you solve it?") 

See notebook and questions for specific questions associated with each one.

1. Data Integration
    - Read CSV
    - Read JSON
    - Read Sqlite
    - Read HTML
    - *Reasoning* 
2. Data Manipulation
    - Integrate parts and all of the data
    - Assertions (correct indices and data types, etc.)
    - *Reasoning*
3. Data Understanding
    - Big picture (characteristics, # of objects, # of missing, etc.)
    - Central Tendency
    - Dispersion
    - Correlation
    - *Reasoning*
4. Data Visualization
    - Relational plots
    - Distributional plots
    - Categorical plots
    - HeatMap
    - *Reasoning*
5. Data Preprocessing
    - Dealing with Missing Values
    - Dealing with Outliers
    - Dealing with Scales (normalizing)
    - Dealing with Continuous Data
    - Dealing with Text {Optional}
    - Save the Processed Data
    - *Reasoning*
6. Survey 
    - (Mainly just a way for the Prof and TA to get some feedback on the exam and course format, etc.)

## HW 5

(Associated with [Classification](Lectures.md#05-data-preprocessing) lecture section)

- [Part A - Reasoning](Assignments/HW5/PartA-Reasoning.html)
    - Discuss the big picture of learning and why we have both supervised and unsupervised learning
    - Someone claims that the "Nearest Neighbor" method is not a learning method. Do you agree or disagree, and why?
    - Why is Logistic Regression considered a classification method?
- [Part B - Application](Assignments/HW5/PartB-Application.html)
    - Classification ML model application to "titanic" dataset
        - Decision tree
        - Logistic regression
        - KNN 
        - SVM
    - Summarize your findings
    - (Optional) [I didn't finish any of them]
        - use other sklearn library model
        - play with other dependent variables
        - use `penguins` dataset and compare performance of classification methods
- [Part C - Learning Resource](Assignments/HW5/PartC-Learning_Resources.html)
    - honestly I have no idea what the original topic(s) were, but it was an assignment that was supposed to find a resource for certain topics

## HW 6

(Associated with [Regression](Lectures.md#07-regression) lecture section)

- [Part A - Reasoning](Assignments/HW6/PartA-Reasoning.html)
    - Can Polynomial regression be compared with multiple regression? Why or why not?
    - Explain the difference between MAE and MSE, and provide scenarios in which one is better than the other.
    - What is R-squared? Why pursuing R-squared may lead to overfitting?
- [Part B - Application](Assignments/HW6/PartB-Application.html)
    - Regression ML model application to "Tetuan-City-power-consumption.csv"
        - Simple Linear Regression
        - Polynomial Linear Regression
        - Polynomial regression with regularized term of choice (Ridge, Lasso, or Elastic Net)
    - Use any combination of independent variables
    - Summarize your findings
- [Part C - Learning Resource](Assignments/HW6/PartC-Learning_Resource.html)
    - Assumptions matter
        - Find a resource that explains the assumptions of linear regression, and demonstrates the consequences if assumptions are not met for the analysis
    - Overfitting
        - Find a comprehensive tutorial that introduces the topic of overfitting and the techniques that help to prevent overfitting
    - (I never answered these)

## HW 7

(Associated with [Clustering](Lectures.md#08-clustering) lecture section)

- [Part A - Reasoning](Assignments/HW7/PartA-Reasoning.html)
    - Identify a real-world business scenario that needs clustering.
    - We learned normalization in data preprocessing. Discuss how scaling impacts some clustering methods, and how it doesn't impact some other clustering methods.
    - We learned that order matters in some classification methods, does order matter in clustering?
- [Part B - Application (Version 1)](Assignments/HW7/PartB-Application.html); [Part B (Version 2: messing around with normalized data)](Assignments/HW7/PartB-Application2.html) 
    - Clustering ML model application to `penguins` dataset
        - K-means
        - hierarchical
        - DBSCAN
            - (I didn't end up doing this one for some reason)
    - PCA analysis 
- [Part C - Learning Resource](Assignments/HW7/PartC-Learning_Resource.html)
    - Curse of Dimensionality
        - Find a resource that explains the Curse of Dimensionality, and demonstrates how high dimensionality affects the clustering methods we learned
    - Agglomerative
        - Find a comprehensive tutorial that introduces the topic of hierarchical clustering and uses Agglomerative method to demonstrate the process.
    - ((Again) I never answered these)

## HW 8

(Associated with [Frequent Patterns](Lectures.md#09-frequent-patterns) lecture section)

- [Part A - Reasoning](Assignments/HW8/PartA-Reasoning.html)
    - Explain why it is impossible to do FP analysis for all items in Walmart
    - Explain each with examples
        - Monotone
        - Antimonotone
        - Convertible monotone
        - Convertible antimonotone
        - Strongly convertible
- [Part B - Application](Assignments/HW8/PartB-Application.html)
    - (Starts from Question 4, because I believe the first three questions were written mathwork for apriori calculation or something)
    - Do Apriori with Python
        - Conduct apriori for 2-itemset association rules mining
        - Conduct apriori for 3 or more itemset association rules mining and...
            - Setup your parameters so you can find ___ rule(s)
    - (Optional) {and I didn't do these}
        - Conduct FP Growth for this dataset and observe the difference
        - Compare the performance of Apriori and FP Growth by controlling the hyperparameters
- [Part C - Learning Resource](Assignments/HW8/PartC-Learning_Resource.html)
    - Constraint Apriori
        - Find a resource that explains the Constraint Apriori, and how the constraints help improve the efficiency
    - Lift
        - Find a comprehensive tutorial that introduces how the concept of Lift is addressed in Association Rules

## [Exam 2](Assignments/JasmineKobayashiExam2/Exam2.html)

1. Read and Understand the data
2. Classification
    - Decision trees
    - Logistic Regression
    - Method of choice
3. Regression
    - Simple Linear Regression
    - Simple Polynomial Regression
    - Multiple Lienar Regression
4. Clustering
    - K-means
    - DBSCAN
    - PCA
5. Frequent Patterns
    - Apriori algorithm to find association rules
6. Outliers (Optional)
7. Neural Network (Optional)
