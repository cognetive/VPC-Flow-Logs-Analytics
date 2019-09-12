
import pandas as pd, numpy as np, matplotlib.pyplot as plt, networkx as nx, tensorflow as tf, seaborn as sn
from geolite2 import geolite2
from scipy import signal
from sklearn.metrics import mean_squared_error, confusion_matrix

# Finds country by IP address.
# Args: ip - String represents an IP address.
# Returns: String represents the corresponding country name.
def get_country(ip):
    try:
        x = geo.get(ip)
    except ValueError:
        return pd.np.nan
    try:
        return x['country']['names']['en'] if x else pd.np.nan
    except KeyError:
        return pd.np.nan

    
# Converts integer to its Metric Prefix (MP) representation.
# Args: num - Integer.
# Returns: A string with the corresponding MP representation.
def number_format(num):
    magnitude = 0
    while num >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.3f%s' % (num, ['', 'K', 'M', 'G', 'T', 'P'][magnitude])


# Iteratively scales an array, obtaining predefined max value and max/min ratio. Useful for illustration tasks. 
# Args: arr - Array of numbers. max_val - the desired max value.
# Returns: A scaled array.
def scale(arr, max_val, ratio=100):
    while np.max(arr) > ratio * np.min(arr):
        arr = np.sqrt(arr)
    arr = arr * max_val / np.max(arr)
    return arr


# Filters according to "Start" column and the given dates.
# Args: df - Dataframe in flow_logs format. (year1, month1, day1) - Start date. (year2, month2, day2) - end date.
# Retruns: A filtered dataframe.
def date_filter(df, year1, month1, day1, year2, month2, day2):
    start_date = pd.to_datetime('%s-%s-%s' % (year1, month1, day1))
    end_date = pd.to_datetime('%s-%s-%s' % (year2, month2, day2))
    mask = (df['Start'] >= start_date) & (df['Start'] <= end_date)
    return df.loc[mask]


## To be removed.
def get_data(kind):
    
    if kind == "Ext_data":
        return ext_data
    
    if kind == "Art_data":
        return art_df.set_index('Start')[['ABBytes']].resample('0.5Min').sum().fillna(0) 
    
    tot_df = flowlogs_df.set_index('Start')[['Total_Packets', 'Total_Bytes']].resample('1H').sum().fillna(0)  

    samples_num = len(tot_df.index)
    sin_arr1 = [np.sin(2*np.pi*x/24) * 0.1 * np.max(tot_df["Total_Packets"]) for x in range(samples_num)]
    sin_arr2 = [np.sin(2*np.pi*x/4) * 0.01 * np.max(tot_df["Total_Packets"]) for x in range(samples_num)]
    tot_df["Noisy_Packets"] = tot_df["Total_Packets"] + sin_arr1 + sin_arr2
    
    tot_df["flows"] = flowlogs_df.set_index('Start').resample('1H').size().replace(0, np.nan)
    tot_df["averageFlowPacket"] = tot_df["Total_Packets"] / tot_df["flows"]
    tot_df["averageFlowByte"] = tot_df["Total_Bytes"] / tot_df["flows"]
    tot_df["averagePacketSize"] = tot_df["Total_Bytes"] / tot_df["Total_Packets"]
    tot_df["flowBehavior"] = tot_df["flows"] / tot_df["averagePacketSize"]
    tot_df = tot_df.fillna(0)

    if kind == "Total_Packets":
        return tot_df.loc[:, ["Total_Packets"]]
    if kind == "Noisy_Packets":
        return tot_df.loc[:, ["Noisy_Packets"]] 
    if kind == "Att":
        return tot_df[["flows", "averageFlowPacket", "averageFlowByte", "averagePacketSize", "flowBehavior"]]

    
# Smoothes dataframe in a column-wise manner using moving average.
# Args: df - Dataframe. w - Window size. s - Stride size.  
# Retruns: Smoothed dataframe.
def smooth_MA(df, w, s=1):
    samples_num = int(len(df.index) / s) - w
    indices = [df.index[i*s+int(0.5*w)] for i in range(samples_num)]
    smoothed_df = pd.DataFrame(index=indices, columns=df.columns)
    for c in df.columns:
        smoothed_df[c] = [np.mean(df[c].values[i:i+w]) for i in samples_num]
    return smoothed_df


# Smoothes dataframe in a column-wise manner using low-pass filter.
# Args: df - Dataframe. coeff - Amount of filter coefficients. thresh - The attenuation frequency. s - Stride size. 
# Retruns: Smoothed dataframe.
def smooth_LP(df, coeff=4, thresh=0.1, s=1):
    samples_num = int(len(df.index) / s)
    indices = [df.index[i*s] for i in range(samples_num)]
    smoothed_df = pd.DataFrame(index=indices, columns=df.columns)
    sample_indices = [i*s for i in range(samples_num)]
    b, a = signal.butter(coeff, thresh)
    for c in df.columns:
        filtered = signal.lfilter(b, a, df[c].values)
        smoothed_df[c] = [filtered[i] for i in sample_indices]
    return smoothed_df


# Normalizes dataframe columns to range [0, 1] using min-max method.
# Args: df - Dataframe.
# Returns: Normalized dataframe.
def normalize(df):
    normalized_df = df.copy()
    for c in normalized_df.columns:
        normalized_df[c] = ((df[c] - np.min(df[c])) / (np.max(df[c]) - np.min(df[c]))).fillna(0)
    return normalized_df


# Standardizes dataframe in a column-wise manner.
# Args: df - Dataframe.
# Returns: Standardized dataframe.
def standardize(df):
    standardized_df = df.copy()
    for c in standardized_df.columns:
        standardized_df[c] = ((df[c] - np.mean(df[c])) / (np.std(df[c]))).fillna(0)
    return standardized_df


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
    
    
# Turns one-dimensional signal into multi-dimensional signal by converting contiguous subsequences to vectors. Useful as a data preperation for time-series prediction.
# Args: df - 1d signal. window - The subsequence size.
# Returns: df_data represents a multi dimensional signal in which row i consists of the [i : i+window] values of the original signal. df_labels is a shift of df by window steps. 
def shingle(df, window):
    df_data, df_labels = pd.DataFrame(index=df.index[window:], columns=range(len(df.columns)*window)), pd.DataFrame(index=df.index[window:], columns=range(len(df.columns)))
    values = df.values.tolist()
    for i in range(len(df_data.index)):
        arr = []
        for k in range(window):
            arr = arr + values[i+k] 
        df_data.iloc[i] = arr
        df_labels.iloc[i] = df.iloc[window+i].values
    return df_data, df_labels


# Plots confusion matrix.
# Args: (a, b) - Binary arrays with the same dimension.
# Returns: None.
def plot_confusion_matrix(a, b):
    cm = confusion_matrix(a, b)
    plt.figure(figsize = (7,7))
    sn.heatmap(cm, annot=True, annot_kws={"size": 12}, cmap='Blues', fmt='g', cbar=False, xticklabels=['False', 'True'], yticklabels=['False', 'True'])
    plt.xlabel('Predicted', fontsize=16), plt.ylabel('Actual', fontsize=16)
    plt.show()

    
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

    
# Predicts time-series values using fully-convolutional neural network.
# Args: (train_set, test_set) - Dataframes with shingled time-series. (train_labels, test_labels) - Dataframes with the next value of the time series. 
#       step - Training rate. max_epochs - Number of training epochs. batch_size - Training batch size. activation - Tensorflow activation function. reg_coeff - Regularization coefficient.
#       tolerance - Amount of consequent epochs in which test loss can increase without stopping the training phase. is_initial - Whether the graph should be restored.
# Returns: Array of the predicted values.
def cnn_regression(train_set, test_set, train_labels, test_labels, step=0.001, max_epochs=100, batch_size=64, activation=tf.nn.tanh, tolerance=0, reg_coeff=0, is_initial=False):
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
            for batch in range(int(train_size / batch_size)):
                sess.run(train, feed_dict={x: train_set.iloc[batch*batch_size:batch*batch_size+batch_size], y: train_labels.iloc[batch*batch_size:batch*batch_size+batch_size]})
            if epoch % 10 == 0:
                print("epoch: %g, train loss: %g" % (epoch, loss.eval(feed_dict={x: train_set, y: train_labels})))
        saver.save(sess, save_path=save_path)
        print("Training finished and saved. Calculating results...")
        pred = np.reshape([x3.eval(feed_dict={x: test_set.iloc[i:i+1], y: test_labels.iloc[i:i+1]}) for i in range(test_size)], len(test_labels.index))
        score = mean_squared_error(pred, np.reshape(test_labels.values, len(test_labels.index)))
        print("Done. Averaged test loss: %f" % score)
    return pred


# Encode multi-dimensional signal using fully-connected neural network.
# Args: train_set, test_set - Dataframes with multi-dimensional signal. encoding_len - The desired encoding size.
#       step - Training rate. max_epochs - Number of training epochs. batch_size - Training batch size. activation - Tensorflow activation function. reg_coeff - Regularization coefficient.
#       tolerance - Amount of consequent epochs in which test loss can increase without stopping the training phase. is_initial - Whether the graph should be restored.
# Returns: Dataframe represents the encoded signal.
def FC_autoencoder(train_set, test_set, encoding_len=None, step=0.001, max_epochs=100, batch_size=64, activation=tf.nn.tanh, tolerance=0, reg_coeff=0, is_initial=False):
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
            for batch in range(int(train_size / batch_size)):
                sess.run(train, feed_dict={x: train_set.iloc[batch*batch_size:batch*batch_size+batch_size]})
            if epoch % 10 == 0:
                print("epoch: %g, train loss: %g" % (epoch, loss.eval(feed_dict={x: train_set})))
        saver.save(sess, save_path=save_path)
        print("Training finished and saved. Calculating results...")
        pred = [x_decoded.eval(feed_dict={x: test_set.iloc[i:i+1]}) for i in range(len(test_set.index))]
        score = mean_squared_error(np.reshape(pred, [len(test_set.index), sample_len]), np.reshape(test_set.values, [len(test_set.index), sample_len]))
        print("Done. Averaged test loss: %f" % score)
        encoded = np.reshape([x_encoded.eval(feed_dict={x: test_set.iloc[i:i+1]}) for i in range(len(test_set.index))], [len(test_set.index), encoding_len])
    return pd.DataFrame(data=encoded, index=test_set.index)

