---
layout: page
title: "STAT 5000: Statistical Methods & Applications I"
permalink: /CUB/StatI
---

[Spring 2023] ***Stat Methods & App. I*** with Professor Abel Iyasele at [CU Boulder](../../CUB.md)

# Individual Project

- [NOAA Data EDA](Project/NOAA-STAT-Project.html)
- [Project code](Project/STAT-Project.html)

# Lecture material (R-code)
Originally the professor provided these to us as R-script files, but I copy and pasted them into (Quarto) Markdown code

## [Visualizations (with `ggplot`)](Lecture-material/Visualization.html)
- Uses "NYC Flights" package (`nycflights13`)
- Scatterplots
    - Address Overplotting
- Line graphs
- Histograms
    - Adjusting bins and binwidth
    - Faceting histograms by variable
- Box plots
- Bar plots
    - cases representing 
        - counts
        - values
    - faceting

## [Variable Moments (shapes of data distributions)](Lecture-material/Moments.html)
- Moments are used to describe the shape of data or a variable.
- Mean
    - Outliers and trimming
- Median
- Mode
    - `modeest` package
- Variability
    - Variance
    - Standard Deviation
- Symmetry
    - Skewness
    - Kurtosis

## [Plotting Normal Curve in R](Lecture-material/NormalDist.html)
- Cumulative density function
- Quantiles of Normal Distributions
- Standard Normal Distribution
- Systematic Sampling
- examples of uses of the following functions
    - `dnorm`,`pnorm`,`qnorm`,`rnorm` (normal distributions)
    - `tigerstats` package (same idea, but also has a parameter to create graph (I think that's only difference))
        - `qnormGC`(etc.)
    - `qt`,`pt`, etc. (t-distributions)

## [Sampling in R](Lecture-material/Sampling.html)
- Sampling (with & without replacement)
- Simple Random Sampling
- Systematic Sampling

## [Sampling Distributions and the Central Limit Theorem](Lecture-material/CLT.html)
- What is the central limit theorem (CLT)
- Key notes about the CLT
- Why the normal distribution is important
- Assumptions of the CLT
- Why is the CLT important?
    - Example

## [Testing Population Means](Lecture-material/TestingPopMean.html)
- One sample Z-testing of population means in R
- Case 1: Z-test for means - population standard deviation is known and sample size is large enough
- Case 2: t-test for means - population standard deviatioon is not known or sample size is small (or both)

## [Two Sample Means Testing](Lecture-material/TwoSampleMeans.html)
- *Independent means T-test:* Unknown population variances but are assumed equal
- *Independent means T-test:* Unknown population variances but are NOT assumed equal
- *Dependent means T-test/Paired Samples T-test*

## [Testing Population Variances](Lecture-material/TestingPopVar.html)
- Ex 1: One-sample one-sided variance test
- Ex 2: One-sample one-sided variance test
- Ex 3: Another one-sided test
- Ex 4: One-sample, two-sided significance test for variance
- Testing two population (variable) variances - the F-ratio
- Testing behavior of proportions in two populations

## [Analysis of Variance (ANOVA)](Lecture-material/ANOVA.html)
- Case 1: Testing hypothesis when samples are independent of each other
    - (2 examples)
- Case 2: ANOVA for dependent samples (i.e. more than two matched samples)
    - (2 examples/cases)

## [Correlation and Regression Analysis](Lecture-material/CorrReg.html)
- Correlation analysis
    - 2 examples
    - types of correlation coefficients (pearson, kendall, spearman)
    - testing hypothesis about correlation measures
    - testing whether two correlation coefficients are the same for two numeric variables
    - correlation matrices
- Regression analysis
    - Simple Linear Regression
        - 1 example
    - Multiple Linear Regression
        - 3 examples
            - example data
            - synthetic data
            - real-world data from web