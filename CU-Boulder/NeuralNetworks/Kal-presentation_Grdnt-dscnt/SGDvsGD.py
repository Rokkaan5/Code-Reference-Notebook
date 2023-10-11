#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 19:14:26 2023

@author: kaloyanparvanov
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def f(x, y):
    return x**2 + y**2 + x*np.sin(5*y) + y*np.sin(5*x)

def gradient(x, y):
    df_dx = 2*x + np.sin(5*y) + 5*y*np.cos(5*x)
    df_dy = 2*y + np.sin(5*x) + 5*x*np.cos(5*y)
    return np.array([df_dx, df_dy])

# Parameters
learning_rate = 0.1
n_iterations = 100

# GD and SGD
start_point = np.array([2.5, 2.5])

theta_gd = np.array(start_point)
theta_sgd = np.array(start_point)

theta_history_gd = [theta_gd]
theta_history_sgd = [theta_sgd]

# Find the near-global minimum within our range
result = minimize(lambda t: f(t[0], t[1]), [0, 0]) # Starting from the point (0,0)
min_point = result.x
min_value = f(min_point[0], min_point[1])

for iteration in range(n_iterations):
    grad_gd = gradient(*theta_gd)
    theta_gd = theta_gd - learning_rate * grad_gd
    theta_history_gd.append(theta_gd)

    # Introduce some noise for SGD to mimic the effect of using a random data point.
    noisy_grad = gradient(*theta_sgd) + np.random.normal(scale=1.5, size=2)
    theta_sgd = theta_sgd - learning_rate * noisy_grad
    theta_history_sgd.append(theta_sgd)

# Plot
x = np.linspace(-3, 3, 300)  # Increased mesh density
y = np.linspace(-3, 3, 300)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

colors = {
    "gd": "limegreen",
    "sgd": "royalblue"
}

for i in range(n_iterations):
    for i in range(n_iterations):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot_surface(X, Y, Z, cmap='cividis', alpha=0.7)  # Changed colormap to viridis
    
        # Plot for GD
        if i > 0:
            ax.plot(*zip(*theta_history_gd[:i+1]), [f(*coords) for coords in theta_history_gd[:i+1]], color=colors["gd"], alpha=0.5)
        ax.scatter(*theta_history_gd[i], f(*theta_history_gd[i]), color=colors["gd"], label='GD', s=60, edgecolors='k')
    
        # Plot for SGD
        if i > 0:
            ax.plot(*zip(*theta_history_sgd[:i+1]), [f(*coords) for coords in theta_history_sgd[:i+1]], color=colors["sgd"], alpha=0.5)
        ax.scatter(*theta_history_sgd[i], f(*theta_history_sgd[i]), color=colors["sgd"], label='SGD', s=60, edgecolors='k')

        # Plot the global min
        ax.scatter(*min_point, min_value, color='red', marker='*', s=200, label='Global Min' if i == 0 else "")
    
        ax.view_init(elev=60, azim=-80)
        ax.legend(loc="upper right")  # Position legend at upper right
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('f(x, y)')
        ax.set_title(f"Iteration {i+1}")
    
        plt.savefig(f"/Users/kaloyanparvanov/Downloads/SGDvsGD/frame_{i}.png")
        plt.close()



