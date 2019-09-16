#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np, matplotlib.pyplot as plt


# Finds extremely-high values within an array.
# Args: scores - Array of values. thresh - Amount of STDs.
# Returns: Array of indices in which the value exceeeds MEAN+thresh*STD.
def find_anomalies(scores, thresh=3):
    threshold = np.mean(scores) + thresh * np.std(scores)
    return [i for (i, x) in enumerate(scores) if x > threshold]


# Marks anomalies in a 1d-signal.
# Args: x - indices of a signal. y - values of a signal. anomaly_indices - Indices in which anomalies occur.
# Returns: None.
def anomaly_visualization(x, y, anomaly_indices):
    plt.figure(figsize=(20,5))
    plt.plot(x, y, 'b')
    for i in anomaly_indices:
        plt.plot(x[i-1:i+1], y[i-1:i+1], 'r', linewidth=2)
    plt.ylim(1.5 * np.min(y), 1.5 * np.max(y))
    plt.title("Signal Anomalies", fontsize=20)
    plt.show()

