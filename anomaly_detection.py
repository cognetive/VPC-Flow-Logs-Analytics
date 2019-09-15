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
# Args: x - indices of a signal. y - values of a signal. anomaly_indices - Indices in which anomalies occur. draw_trends - Whether to split the graph into segments.
# Returns: None.
def anomaly_visualization(x, y, anomaly_indices, draw_trends=False):
    plt.figure(figsize=(20,5))
    if draw_trends == False:
        plt.plot(x, y, 'b')
        for i in anomaly_indices:
            plt.plot(x[i-1:i+1], y[i-1:i+1], 'r', linewidth=2)
    else:
        anomaly_indices = anomaly_indices + [0, len(x)-1]
        anomaly_indices.sort()        
        for i in range(1, len(anomaly_indices)):
            start, end = anomaly_indices[i-1], anomaly_indices[i]
            mid = int(0.5 * (start + end - 1))
            plt.plot(x[start:end], y[start:end])
            plt.annotate("Avg: %.2f" % np.mean(y[start:end]), xy=(x[mid], 0.1*np.max(y) + y[mid]), ha='center', bbox=dict(boxstyle="round4", fc="1."))
            if i < len(anomaly_indices) - 1:
                plt.plot(x[end-1:end+1], y[end-1:end+1], 'r', dashes=[3, 3])
    plt.xlabel('Time', fontsize=20), plt.ylabel('Packets', fontsize=20)
    plt.ylim(1.5 * np.min(y), 1.5 * np.max(y))
    plt.title("Signal Anomalies", fontsize=20)
    plt.show()

