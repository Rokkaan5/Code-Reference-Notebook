#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 18:57:15 2023

@author: kaloyanparvanov
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier


# Create synthetic data
X, y = make_classification(n_samples=100, n_features=1, n_informative=1, n_redundant=0, n_clusters_per_class=1, random_state=42)

# Parameters
learning_rate = 0.01
batch_size = 10  # Adjusted for the given dataset size
n_epochs = 10
n_batches = int(X.shape[0] / batch_size)

# Initialize lists for storing plots
losses = []
weights = []
weights_0 = []

# Initialize the model
clf = SGDClassifier(loss="log", learning_rate="constant", eta0=learning_rate, penalty=None, max_iter=1, average=False, warm_start=True, fit_intercept=True)

# Create an array of indices for shuffling
indices = np.arange(X.shape[0])

for epoch in range(n_epochs):
    # Shuffle the indices at the beginning of each epoch
    np.random.shuffle(indices)

    for i in range(n_batches):
        batch_indices = indices[i * batch_size: (i + 1) * batch_size]
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        
        # Fit the model on the current batch
        clf.partial_fit(X_batch, y_batch, classes=[0, 1])
        
        # Calculate the probability for the sigmoid
        prob = clf.predict_proba(X)[:, 1]
        
        # Store loss and weights
        loss = -np.mean(y * np.log(prob + 1e-9) + (1 - y) * np.log(1 - prob + 1e-9))
        losses.append(loss)
        weights.append(clf.coef_[0][0])
        weights_0.append(clf.intercept_[0])

        # Plotting
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Sigmoid and data points
        axs[0].scatter(X, y, c='gray', alpha=0.2)
        axs[0].scatter(X_batch, y_batch, c=y_batch, cmap='rainbow', alpha=0.9)
        sorted_order = np.argsort(X[:, 0])
        axs[0].plot(X[sorted_order, 0], prob[sorted_order], color="black")
        axs[0].set_title("Fitted Sigmoid & Data Points")
        axs[0].set_xlabel("Feature Value")
        axs[0].set_ylabel("Probability")

        # Plot 2: Loss
        axs[1].plot(losses)
        axs[1].set_title("Loss over Iterations")
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel("Log Loss")

        # Plot 3: Weights
        axs[2].plot(weights, label="Feature Weight")
        axs[2].plot(weights_0, label="Intercept")
        axs[2].legend()
        axs[2].set_title("Weights over Iterations")
        axs[2].set_xlabel("Iteration")
        axs[2].set_ylabel("Weight Value")

        plt.tight_layout()
        plt.savefig(f"/Users/kaloyanparvanov/Downloads/SGD_Logistic_1sample_gif/iteration_{epoch * n_batches + i}.png")
        plt.close()

print("Visualization saved!")

