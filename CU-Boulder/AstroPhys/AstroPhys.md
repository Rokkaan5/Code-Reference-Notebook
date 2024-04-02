---
layout: page
title: "ASTR 5550: Observations, Data Analysis, and Statistics"
permalink: /CUB/AstroPhys
---

[Spring 2024] ***Observations, Data Analysis, and Statistics*** with Dr. Robert Ergun at [CU Boulder](../../CUB.md)

---

My last elective course for my Master's Data Science degree within in the Astrophysics department.


# Homework

Any parts of any of the hw that contained coding portions

## Helper Functions
- [Notebook version of hw-helper functions](HW/hw_helper_func2.html)

## HW1
- [HW1 - Question 3: Binomial and Poisson Distributions](HW/hw1/hw1.html)
    - Mainly just comparing Binomial vs. Poisson Distribution 
        -(like their difference when *p* is not small enough)

## HW2
- [HW2 (clean)](HW/hw2/hw2-clean.html); [HW2 (clean & "fixed")](HW/hw2/hw2-clean-fixed.html)
    1. Binomial and Gaussian Distributions
        - Compare Binomial vs. Poisson vs. Gaussian
    2. How Old is This Volcano?
        - Age estimation based on count rate example
        - Posterior probability (Bayes law)
            - using Poisson as parent distribution
        - various versions of sigma calculations
            - simple
            - distribution
            - FWHM
    3. Random Wak and Central Limit Theorem
        - Drunk walk example
            - Compare Binomial and Gaussian
            - Drunk walk with slopes
                - bottom of valley
                - top hill
    4. Not So Random Walk: Power Law
        - Step size changes based on location


## HW3
- [HW3](HW/hw3/hw3.html)
    1. Joint Probability and Find the Gap
        - How many FOV for greater than 50% probability of seeing 100 galaxies 
        - How many for greater than 99%
        - Contaminating stars: how many sectors to find enough uncontaminated FOVs?
        - "Mind the Gap" lecture application
    2. Multi-variant Uncertainties
        - What is the 1-sigma variance?
        - What is the uncertainty?
        - Which uncertainty makes the largest contribution to the overall uncertainty?
    3. Gamma Ray Bursts
        - Estimate (no computer) how long was it expected to take the observations to show that the fraction of events in teh galactic plane is 2-sigma higher than 50% proving a galactic origin model?
            - (Solutions = good example how using Binomial is easier solution, but can get same solution with Poisson )
        - Compute CDF, what is the probability that ___ is 60% or higher?
        - Does given histogram (H = [...]) match a Poisson distribution?
            - What is the (binned) chi-squared?
            - How does the chi-squared compare with the expected distribution?
        - Explode H into a 300-point array, and perform another chi-squared test.

## HW4
- [HW4](HW/hw4/hw4.html)

## HW5
- [HW5](HW/hw5/hw5.html)