#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd, numpy as np, tensorflow as tf
from geolite2 import geolite2

# Finds country by IP address.
# Args: ip - String represents an IP address.
# Returns: String represents the corresponding country name.
def get_country(ip, geo):
    try:
        x = geo.get(ip)
    except ValueError:
        return pd.np.nan
    try:
        return x['country']['names']['en'] if x else pd.np.nan
    except KeyError:
        return pd.np.nan


# Add source and destination countries to a Dataframe.
# Args: df - Dataframe that contains src_ip and dst_ip columns.
# Returns: Extended Dataframe contaning columns for the source and destination countries.
def add_countries(df):
    print("This might take a while, please wait...")
    geo = geolite2.reader()
    df['src_country'] = df['src_ip'].apply(get_country, geo=geo)
    df['dst_country'] = df['dst_ip'].apply(get_country, geo=geo)
    geolite2.close()
    return df


# Converts integer to its Metric Prefix (MP) representation.
# Args: num - Integer.
# Returns: A string with the corresponding MP representation.
def number_format(num):
    magnitude = 0
    while num >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.3f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])

    
# Caclculates convolution layer.
# Args: x - Signal. W - Convolution kernel. stride - Convolution stride.
# Returns: Convolution layer.
def conv2d(x, W, stride=1):
    return tf.nn.conv2d(x, W, strides=stride, padding='SAME')


# Caclculates pooling layer.
# Args: x - Signal. stride - Pooling size.
# Returns: Pooling layer.
def max_pool(x, stride=2):
    return tf.nn.max_pool(x, ksize=[1, 1, stride, 1], strides=[1, 1, stride, 1], padding='SAME')


# Initializes weight variables using xavier method.
# Args: kind - layer kind (FC / conv). (size_in, size_out) - dimensiones for FC layer.
# Returns: Initialized weight variables.
def get_variables(kind, size_in=None, size_out=None):
    initializer = tf.contrib.layers.xavier_initializer()
    if kind == "FC":
        return tf.Variable(initializer([size_in, size_out])), tf.Variable(initializer([size_out]))
    elif kind == "conv":
        return tf.Variable(initializer([1, 5, 1, 1])), tf.Variable(initializer([1]))


# Builds Tensorflow graph.
# Args: sess - Tensorflow session. saver - Tensorflow saver. init - Tensorflow initializer. is_initial - Whether the graph should be restored. path - Location of the graph to be restored.
# Returns: None.
def build_model(sess, saver, init, is_initial, path):
    if is_initial==True:
        sess.run(init)
        print("Model initialized.")
    else:
        saver.restore(sess, save_path=path)
        print("Model restored.")
    print("Start training...")

