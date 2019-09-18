# Anomaly Detection 
This tutorial demonstrates how to use Analytics to find anomalies in VPC Flow Logs.   
If you are still not familiar with Analytics, consider starting with the *hello world* tutorial.

## Preparing the Environment
1. Load *anomalies_cnn* into an IPython notebook running with Python 3.
2. Install *Analytics* package (if needed).
3. Download *data_example.csv* to your local machine.
4. Upload the data into your notebook as a Pandas DataFrame named *flowlogs_df*.

## Let's Get Started
### Selecting the Data
First we need to extract and aggregate the specific metric (column) in which we look for anomalies:

![alt text](images/metric_agg.png)

![alt text](images/time_series_1d.png)  

<br/>

### Preprocessing
Our raw data is ready. But before feeding the time-series predictor, we should split the data into train and test set; After that we can standardize and shingle each set:

![alt text](images/split_shingle.png)  

![alt text](images/prepare_data.png)

We can verify that our data is properly standardized and shingled:
![alt text](images/shingle_verify.png)  

Great! Our data is ready for the learning task.

<br/>

### Learning and Predicting
Now we can run our CNN predictor, provided with the data and the desired learning duration:

![alt text](images/epochs.png)  

![alt text](images/learning_1.png)  

Note that we got a test error of 1.06. We can try to improve that by extending the learning phase; We pass **is_initial=False** to continue the learning from the stopping point:

![alt text](images/learning_2.png)  

Indeed, we got a better error now. We are ready to visualize the anomalies!  

<br/>

### Visualizing the Results

First, we set an *anomaly score* for each time by calculating the distance between the predicted and the real values.   
Next, we find extremely-high scores by passing the anomaly scores to **find_anomalies()**.   
Last, we plot the results by calling to **anomaly_visualization()** (anomalies will be marked by *red*).  
We can execute the whole process by running the last cell of this notebook:

![alt text](images/visualization.png)

Nice! As we could expect, the large splikes are spotted as anomalies. We can decide when a score is high enough to be considered as anomalous, by passing an explicit threshold to find_anomalies(). For example, let's set **threshold = &mu;+8&sigma;**:

![alt text](images/visualization_2.png)

<br/>
<br/>

That's it! In this tutorial we learned how to use Analytics module for anomaly detection in 1d time-series. More information regarding this module capabilities can be found in the project's documentation. Hope you found this tutorial useful!
