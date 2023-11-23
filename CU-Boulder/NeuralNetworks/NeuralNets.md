---
layout: page
title: "CSCI 5922: Neural Networks & Deep Learning"
permalink: /CUB/NeuralNetworks
---

[Fall 2023] ***Neural Networks*** with Dr. Ami Gates at [CU Boulder](../../CUB.md)

# Sub pages

- [Professor Gates' Code Examples](Prof-Code-Ex.md)
- [Jasmine's Homework Submissions](JK-HW.md)

# Module 1: Intro to NN and Logistic Regression

## Resources
[Two code examples by professor:](Prof-Code-Ex.md#module-1) 

- Simple NN with Sklearn
- Logistic Regression

[One Assignment:](JK-HW.md#module-1) 

- Assignment 1

**Extra Codes for Gradient Descent by Kal Parvanov**

(all raw code)

- [1d Gradient Descent](kal-presentation_Grdnt-dscnt/1d_GD-raw.html)
- [Stochastic vs. Gradient Descent](kal-presentation_Grdnt-dscnt/SGDvsGD-raw.html)
- [Logistic Minibatch](kal-presentation_Grdnt-dscnt/Logistic_minibatch-raw.html)

## Topics
- Introduction to Neural Networks
    1. History
    2. Challenges
    3. Applications
    4. Current Focus
    5. Discussion and Ethics

- Brief Review of Machine Learning Concepts, Data Formats, and Modeling Preparation Requirements for Data and Labels
    1. Review of concepts in modeling data for prediction/classification
    2. Gathering data and related challenges
    3. Formatting, cleaning, and preparing data for supervised learning
        - About Keras and Getting TF/Keras working in Python ([Keras about page](https://keras.io/about/))

- Logistic Regression in Python and by hand
    - Loss function (Cross Entropy)
    - Activation function (Sigmoid)
    - Gradient Descent - Chain Rule - Partial Derivatives
        - Updating parameters
    - Visualizing changes in the Loss

# Module 2: Neural Networks

## Resources
[Three code examples by professor (+ one extra):](Prof-Code-Ex.md#module-2) 

- NN to predict XOR 
- NN with 3D Data - Binary Label
- Using FF, BP, Softmax, CCE, One-Hot Encoding, etc.
- ***Extra:*** Softmax, CCE, and One-Hot Encoding - 3 Outputs

[One Assignment - Two Parts:](JK-HW.md#module-2) 

- Assignment 2 - Part 1: One Layer NN with One Output
- Assignment 2 - Part 2: Multinomial NN with Softmax, CCE, and One-Hot Encoding

## Topics

1. Architectures
2. Derivatives and Gradient Descent
3. Activation Functions and their Derivatives
    - Sigmoid
    - Softmax
    - reLU
4. Loss Functions and their Derivatives
    - MSE
    - Categorical Cross Entropy (CCE)
5. Label Encoding - One-Hot Encoding (OHE)
6. Back Prop (BP) for one hidden layer - one output - ANN
7. Back Prop for NN with multinomial output, one-hot encoded labels, multiple activation functions, CCE Loss Functions
    - Relative derivatives and proofs
8. Coding the XOR Problem using NN (one hidden layer perceptron)
9. Coding a multinomial NN with one-hot encoding, CCE, softmax, FF, BP
10. Related Math and Proofs, Jacobian Matrices, etc.
11. All coding will be "by hand" in Python

# Module 3

CNNs and RNNs

- Introduction to Convolutional Neural Networks (CNNs)
    1. Image analysis and formats review
        a. Image formats and formatting
            i. RGB, GS, 2D, 3D, etc.
            ii. Pixels and image representations
            iii. Reading and flattening images into Python
            iv. Image formats required for TF/Keras
        b. Images as 2D sequential data - concepts - similarities/differences to text, spatial information, etc.
    2. CNN Elements (using TF/Keras)
        a. Kernels and filters
        b. Convolution and cross-correlation
        c. Derivatives and Back Prop
        d. Common Loss and Activation Functions
        e. Pooling (such as max pooling)
        f. Layer options
    3. CNNs with TF/Keras
        a. Layer options
        b. Training and Testing
        c. Accuracy measures and illustrations
    4. Dense/Flattened Layers
    5. Applications and Examples
    6. CNNs vs. ANNs
- Recurrent Neural Networks (RNNs) and Long Short Term Memory (LSTM)
    1. Review of concepts of recurrence relations and recursion
    2. Architecture of RNNs
    3. Sequential data and RNNs
    4. Comparison between ANN, CNN, and RNN
    5. Limitations of RNN
        a. Exploding and vanishing gradients
    6. Derivatives and Back Prop for RNNs (and examples of the gradient issue cause)
    7. History of RNNs and LSTM (Long Short Term Memory) RNNs
    8. Coding RNNs and LSTM RNNs in Python with TF/Keras

Extras:

- Translators
    - LSTM Encoder/Decoder Language Translation
    - Spanish-English LSTM Keras Translator
- Embedding and Sentiment Analysis
- Attention and Transformers (Introduction ONLY)
- Using Tensorflow/Keras - 2D IMages and CNNs
- RNN/LSTM Using TF/Keras and MNIST Data


## Resources
[Various code examples by professor:](Prof-Code-Ex.md#module-3) 

- General Keras
    - Intro to Keras 
    - Simple Examples of ANNs, RNNs (and LSTM, BRNN), and CNNs in Python/Keras 
    - TF-Keras with record data 
- Convolutional Neural Networks (CNNs)
    - CNN with Conv2D Image data in Keras 
    - CNN with Conv2D with CIFAR10 data - Image prediction
- Recurrent Neural Networks (RNNs) and Long Short Term Memory (LSTM)
    - RNN with MNIST 
- Extras
    - English/Spanish Translator with LSTM Transformer 
    - Advanced Sentiment Analysis with Glove Embedding 

[One Assignment - Five parts:](JK-HW.md#module-3) 