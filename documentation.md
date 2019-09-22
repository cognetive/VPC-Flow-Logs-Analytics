## Anomaly Detection Module
This module provides two models to detect anomalies in flow logs, both are based on neural networks.
### Detect Anomalies with Fully Convolutional Network
This network is designed to detect *contextual anomalies* in 1d time series. Given k-consequtive values of a time series,
the network tries to predict the next value. Anomalies can be defined as the values which the network fails to predict.
#### Structure
This model consists of a (potentially deep) neural network, which consists of 4 layers: Input, output, convolution layer 
and fully-connected layer. The convolution is performed in the temporal axis, relying on the existence of temporal patterns in the input.
#### How to Use
1. Use *shingle()* to create the data set and the labels set. Each data point consists of k-consequtive values of the time series,
and each label is the next value of the time series.
2. Split each set into train and test sets, and feed the network.
3. Asign anomaly score for each instance (the distance between the predicted and the real value, according to some metric).
4. Define the instances that received the highest scores as anomalous.
