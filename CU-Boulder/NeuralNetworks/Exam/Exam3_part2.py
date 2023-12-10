# %% libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os,sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers

# %% Question 2 - Part (a)
My_NN_Model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(4,input_shape=(4,1),activation='linear'), 
    tf.keras.layers.Dense(4, activation='sigmoid'),
    tf.keras.layers.Dense(3, activation='relu'), 
    tf.keras.layers.Dense(3, activation='softmax')
  
])

# %% Question 2 - Part (b)
My_CNN_Model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(2, 
                           input_shape=(30,30,1),
                           kernel_size=3,
                           strides=(1,1),
                           padding="same",
                           activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(3, activation='softmax')
])

# %%
