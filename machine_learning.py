#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd, numpy as np, tensorflow as tf
from sklearn.metrics import mean_squared_error
from utilities import conv2d, max_pool, get_variables, build_model


# Predicts time-series values using fully-convolutional neural network.
# Args: (train_set, test_set) - DataFrames with shingled time-series. (train_labels, test_labels) - DataFrames with the next value of the time series. 
#       step - Training rate. max_epochs - Number of training epochs. batch_size - Training batch size. activation - Tensorflow activation function. reg_coeff - Regularization coefficient.
#       tolerance - Amount of consequent epochs in which test loss can increase without stopping the training phase. is_initial - Whether the graph should be restored.
# Returns: Array of the predicted values.
def cnn_regression(train_set, test_set, train_labels, test_labels, step=0.001, max_epochs=100, batch_size=64, activation=tf.nn.tanh, tolerance=0, reg_coeff=0, is_initial=True):
    tf.reset_default_graph() 
    sample_len = len(train_set.columns)
    W1, b1 = get_variables(kind="conv")
    W2, b2 = get_variables(kind="FC", size_in=int(sample_len/2), size_out=int(sample_len/4))
    W3, b3 = get_variables(kind="FC", size_in=int(sample_len/4), size_out=1)
    reg_term = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3)
    x, y = tf.placeholder(tf.float32, [None, sample_len]), tf.placeholder(tf.float32, [None, 1])
    x_reshaped = tf.reshape(x, [-1, 1, sample_len, 1])
    x1 = activation(max_pool(conv2d(x_reshaped, W1, stride = 1) + b1))
    x1 = tf.reshape(x1, [-1, int(sample_len/2)])
    x2 = activation(tf.matmul(x1, W2) + b2)
    x3 = tf.matmul(x2, W3) + b3
    loss = tf.reduce_mean(tf.square(x3 - y)) + reg_coeff*reg_term
    train = tf.train.RMSPropOptimizer(step).minimize(loss, var_list=[W1, W2, W3, b1, b2, b3])
    saver = tf.train.Saver()
    save_path='path'
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        build_model(sess, saver, init, is_initial, save_path)
        test_loss, count = np.inf, 0
        for epoch in range(max_epochs):
            new_test_loss = loss.eval(feed_dict={x: test_set, y: test_labels})
            count = count+1 if new_test_loss > test_loss else 0
            if count > tolerance:
                print("Overfitting, early stopping...")
                break
            test_loss = new_test_loss
            for batch in range(int(len(train_set.index) / batch_size)):
                sess.run(train, feed_dict={x: train_set.iloc[batch*batch_size:batch*batch_size+batch_size], y: train_labels.iloc[batch*batch_size:batch*batch_size+batch_size]})
            # Uncomment to track train loss
            # if epoch % 10 == 0:
                # print("epoch: %g, train loss: %g" % (epoch, loss.eval(feed_dict={x: train_set, y: train_labels})))
        saver.save(sess, save_path=save_path)
        print("Training finished and saved. Calculating results...")
        pred = np.reshape(x3.eval(feed_dict={x: test_set, y: test_labels}), len(test_labels.index))
        score = mean_squared_error(pred, np.reshape(test_labels.values, len(test_labels.index)))
        print("Done. Averaged test loss: %f" % score)
    return pred


# Encode multi-dimensional signal using fully-connected neural network.
# Args: train_set, test_set - DataFrames with multi-dimensional signal. encoding_len - The desired encoding size.
#       step - Training rate. max_epochs - Number of training epochs. batch_size - Training batch size. activation - Tensorflow activation function. reg_coeff - Regularization coefficient.
#       tolerance - Amount of consequent epochs in which test loss can increase without stopping the training phase. is_initial - Whether the graph should be restored.
# Returns: DataFrames represent the encoded and the reconstructed signals.
def FC_autoencoder(train_set, test_set, encoding_len=None, step=0.001, max_epochs=100, batch_size=64, activation=tf.nn.tanh, tolerance=0, reg_coeff=0, is_initial=True):
    tf.reset_default_graph() 
    sample_len = len(train_set.columns)
    if encoding_len == None:
        encoding_len = int(np.sqrt(sample_len))
    W1, b1 = get_variables(kind="FC", size_in=sample_len, size_out=encoding_len)
    W2, b2 = get_variables(kind="FC", size_in=encoding_len, size_out=sample_len)
    reg_term = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)
    x = tf.placeholder(tf.float32, [None, sample_len])
    x_encoded = activation(tf.matmul(x, W1) + b1)
    x_decoded = tf.matmul(x_encoded, W2) + b2
    loss = tf.reduce_mean(tf.square(x_decoded - x)) + reg_coeff*reg_term
    train = tf.train.RMSPropOptimizer(step).minimize(loss, var_list=[W1, W2, b1, b2])
    saver = tf.train.Saver()
    save_path='path'
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        build_model(sess, saver, init, is_initial, save_path)
        test_loss, count = np.inf, 0
        for epoch in range(max_epochs):
            new_test_loss = loss.eval(feed_dict={x: test_set})
            count = count+1 if new_test_loss > test_loss else 0
            if count > tolerance:
                print("Overfitting, early stopping...")
                break
            test_loss = new_test_loss
            for batch in range(int(len(train_set.index) / batch_size)):
                sess.run(train, feed_dict={x: train_set.iloc[batch*batch_size:batch*batch_size+batch_size]})
            # Uncomment to track train loss
            # if epoch % 10 == 0:
                # print("epoch: %g, train loss: %g" % (epoch, loss.eval(feed_dict={x: train_set})))
        saver.save(sess, save_path=save_path)
        print("Training finished and saved. Calculating results...")
        pred = np.reshape(x_decoded.eval(feed_dict={x: test_set}), [len(test_set.index), sample_len])
        score = mean_squared_error(pred, np.reshape(test_set.values, [len(test_set.index), sample_len]))
        print("Done. Averaged test loss: %f" % score)
        encoded = np.reshape(x_encoded.eval(feed_dict={x: test_set}), [len(test_set.index), encoding_len])
    return pd.DataFrame(data=encoded, index=test_set.index), pd.DataFrame(data=pred, index=test_set.index, columns=test_set.columns)

