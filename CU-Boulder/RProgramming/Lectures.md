---
layout: page
title: "R-Programming Lecture Codes"
permalink: /CUB/RProgramming/lecture_codes
---

Lecture codes for [R-Programming course](RProgramming.md) at [CU-Boulder](../../CUB.md)

---

Some of the code might be broken depending on if it's pre-class or post-class codes. 
And some of the codes didn't have html files made yet, and I haven't taken the time to create them again.


(So might need to fix later)

# [Code Day1](Lectures\Code-day1.html) 
- (originally just an R-script file)

# [Programming Basics](Lectures\Programming_Basics.html)
- Basics of markdown and adding R code chunks
- Looks at sample dataframe `murders` dataset from `dslabs`
- some simple `if`/`else` and `print()` statements

# [Functions and loops](Lectures\FUnctions_and_loops_basics.html)
- use of `ifelse()` function (as opposed to separate `if` & `else`)
- example use of `na_example` dataset from `dslabs`
- **Defining Functions**
- **For Loops**
    - Simple scatter plot using base R (`plot()`)
- **Vectorization and Functionals**
    - Vectorization over loops shortens & clears up code
        - Vectorized function = function that applies same operation on each vector
    - Functionals are functions that help apply the same function to each entry in a vector, matrix, data frame or a list.
    - example use of `sapply()`
        - `sapply()` = element-wise operations on any function

# [Quick Note: How to use dollar-operator](Lectures\Quick_notes.html)
- short code showing example use of dollar operator (`$`) to select column in a dataframe.

# [Introduction to Tidyverse](Lectures\Introduction-to-the-Tidyvrse-Class-Lecture.html)
- **What is the Tidyverse?**
- **Some (Very) Basic Plotting with `ggplot`**
    - using `data(mpg)` dataset
    - basic scatter plot 
    - boxplot w/ colors (`geom_boxplot()`)
    - colored scatter plot (`geom_point()`)
    - `facet_wrap()`
    - `geom_smooth()`
    - layer different types of plots in a single plot
        - `geom_smooth() + geom_point()`
- **Data Manipulation and Transformation**
    - libraries
        - `gapminder` (dataset)
        - `dplyr`
    - filter rows with `filter()`
        - `arrange()` to indicate order of data (ascending or descending)
    - select columns with `select()`
    - changing columns with `mutate()`

# [Data Wrangling](Lectures/Post-class_codes/Data-Wrangling-Post-Class-Codes.html)
- **Tidy data**
- using `gapminder` data 
- use of `%in%` operator
    - this operator checks two vectors to see if there are any overlapping numbers
- simple colored scatter plot of `gapminder` data using `ggplot`
- "wide" data vs. "tidy" data

# [Combining Tables](Lectures\Lecture-Combining-Tables.html)
- libraries
    - `tidyverse`
    - `ggrepel`
    - `dslabs`
- data (from `dslabs`): 
    - `murders`
    - `polls_us_election_2016` (also works as `results_us_election_2016`, I think???)
- **Joins**
    - `join` functions in `dplyr`
        - `left_join`
            - plots 
                - scatter (`geom_point()`) 
                - line (`geom_smooth()`) 
                - log transformed plot with line and scatter plots combined
                    - `geom_point()`
                    - `geom_text_repel()`
                    - `scale_x_continuous()` (and also for `y`)
                    - `geom_smooth(method='lm',se = FALSE)`
                    - labelled axes using `xlab()` and `ylab()`
        - `slice()` (to create subsets of data I think)
        - `right_join()`
        - `inner_join()`
        - `full_join()`
        - `semi_join()`
        - `anti_join()`
- **Binding (varies for both Base R and `Tidyverse`)**
    - **Columns**
        - `bind_cols()` function (by `dplyr`)
            - makes columns into a `tibble` (I guess)
            - can also bind dataframes
    - **Rows**
        - `bind_rows()`
- **Set Operators (unions, intersections of sets)**
    - `tidyverse`(or more specifically `dplyr`) loaded, we can apply these operators on dataframes (not just vectors)
    - **Intersect:** `intersect()`
    - **Union:** `union()`
    - **Set Difference:** `setdiff()`
    - **Set Equal:** `setequal()`

# [Reshaping Data](Lectures\Post-class_codes\Reshaping_Data_Post_Class_Codes.html)
- libraries: 
    - `tidyverse`
    - `dslabs`
- data
    - `dslabs`: "fertility-two-countries_example.csv"
    - `gapminder`
- `gather()` (converts wide data into tidy data)
    - example use of `class()` (to show "class" (or data type) of something: integer, character, etc.)
- `spread()` (basically inverse of `gather()`, in other words: turns tidy data into wide data)
- `separate()` 
    - example usefulness using data that starts as wide data, but not exactly clean after using `gather()`
- `unite()` (inverse of `separate()`)

# [Webscraping](Lectures\Lecture-6-webscraping-Pre-Class-Codes.html)
- **The `rvest` package**
    - (within `tidyverse` package)
    - a web harvesting package
    - example use of `read_html(url)`
        - `html_nodes()`
        - extracting tables from html pages like wikipedia
- **CSS Selectors**
    - CSS = "Cascading Style Sheets"
    - `get_recipe(url)`

# [Gapminder Case Study](Lectures\Lecture-7-Gapminder-Pre-Class-Codes.html)
- Data visualization case study
- Trends in world health and economics
- *LOTS* of plots

# [Random Variables](Lectures\Random-Variables-Pre-Class-Codes.html)
- example uses of 
    - `rep()`
    - `sample()` 
- **Sampling Models**
    -**Probability distribution of a random variable**
        - Ex: running a monte carlo simulation
        - function building
        - `replicate()` (runs a function an indicated number of times)
        - `mean()`
        - `sd()`
        - `pbinom()` (and `dbinom`)
    - **Distribution versus probability distributions**
    - **Notation for Random Variables**
- **Central Limit Theorem**
- **Expected Value and Standard Error**
- **Central Limit Theorem Approximation**
    - **Averages and proportions**
    - **Law of large numbers**
    - **Misinterpreting law of averages**
    - **How large is large in the CLT?**
    - **CLT Video**
- **SD versus estimate of SD**
    - `library(dslabs)`
    - `identical()`

---

The following machine learning codes can only run if the previous ones were run before hand (libraries, fitted models, etc.):

- Machine Learning Day1
- Machine Learning Day2
- Machine Learning Day3
- Decision Trees

(None of the above code related to ML is in the repo so far (Aug 2023) because I can't get the lecture code to work and thus knit)