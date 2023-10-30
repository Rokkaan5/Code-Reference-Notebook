---
layout: page
title: "Neural Networks: Prof Gates' Code Examples"
permalink: /CUB/NeuralNetworks/ProfGatesCode
---
# [CU Boulder](../../CUB.md): [Neural Networks](NeuralNets.md)
(Fall 2023) Neural Networks with Dr. Ami Gates at CU Boulder

## Dr. Gates' Code Examples

Most (if not all) code made availabe through her own [website](https://gatesboltonanalytics.com/)

- Reference: Professor Ami Gates, Dept. Applied Math, Data Science, University of Colorado
- All pages of raw-code = unedited code as obtained from Dr. Gates' website
- All markdown versions = versions of Dr. Gates' code cleaned up into a notebook or markdown of somesort (Jupyter or Quarto) edited by me (Jasmine) to show code output, etc.

# Module 1

- Simple NN using just Sklearn ([Raw code](raw_code/Module1/NN_Just_sklearn_Gates.html); [Markdown](mrkdwn/Module1/NN_Just_sklearn_Gates.html))
- Logistic Regression ([Raw code](raw_code/Module1/LogReg_gates.html); [Markdown](mrkdwn/Module1/LogReg_gates.html))
    - Using gradient descent
    - Updating **w** and with GD and LR
    - Using the Loss Function LCE: Cross Entropy
    - Using the Sigmoid as the logistic activation
    - Vis

# Module 2

- Using FF and BP to Train a NN to predict XOR ([Raw code](raw_code/Module2/XOR_NN.html); [Markdown](mrkdwn/Module2/XOR_NN.html))
- Using FF and BP - NN - 3D Data - Binary Label ([Raw code](raw_code/Module2/Mod2_3D_binary.html); [Markdown](mrkdwn/Module2/Mod2_3D_binary.html))
- Using FF, BP, Softmax, CCE, One-Hot Encoding, etc. ([Raw code](raw_code/Module2/Multinomial_NN.html); [Markdown](mrkdwn/Module2/Multinomial_NN.html))

## Extra Resources

- Softmax, CCE, and One-Hot Encoding - 3 Outputs ([Raw code](raw_code/Module2/Extra-3Outputs.html); [Markdown](mrkdwn/Module2/Extra-3Outputs.html))
- Gates' Webpage for all extra resources for this module [here](https://gatesboltonanalytics.com/?page_id=680)


# Module 3

- Intro to Keras ([Raw code](raw_code/Module3/intro_to_keras.html); markdown)
- Simple Examples of ANNs, RNNs (and LSTM, BRNN), and CNNs in Python/Keras ([Raw code](raw_code/Module3/simple_ex_all.html);markdown)


## Convolutional Neural Networks (CNNs)

- Simple Example (from comprehensive code) (markdown section here)
- CNN with Conv2D Image data in Keras ([Raw code](raw_code/Module3/CNN/Conv2D.html); markdown)
- CNN with Conv2D with CIFAR10 data - Image prediction ([Raw code](raw_code/Module3/CNN/CIFAR10_image_pred.html); markdown)

## Recurrent Neural Networks (RNNs) and Long Short Term Memory (LSTM)
1. Review of concepts of recurrence relations and recursion
2. Architecture of RNNs
3. Sequential data and RNNs
4. Comparison between ANN, CNN, and RNN
5. Limitations of RNN
    a. Exploding and vanishing gradients
6. Derivatives and Back Prop for RNNs (and examples of the gradient issue cause)
7. History of RNNs and LSTM (Long Short Term Memory) RNNs
8. Coding RNNs and LSTM RNNs in Python with TF/Keras

## Extras:

- Translators
    - LSTM Encoder/Decoder Language Translation
    - Spanish-English LSTM Keras Translator
- Embedding and Sentiment Analysis
- Attention and Transformers (Introduction ONLY)
- Using Tensorflow/Keras - 2D IMages and CNNs
- RNN/LSTM Using TF/Keras and MNIST Data
