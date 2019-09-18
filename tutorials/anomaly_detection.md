## Anomaly Detection 
This tutorial demonstrates how to use Analytics to find anomalies in VPC Flow Logs. 
If you are still not familiar with Analytics, consider starting with the *hello world* tutorial.

### Preparing the Environment
1. Load *anomalies_cnn* into an IPython notebook running with Python 3.
2. Install Analytics (if needed).
3. Load the data into a Pandas DataFrame named *flowlogs_df*.

### Let's Start
First we set the metric in which we look for anomalies and the aggregation interval length:  

![alt text](images/metric_agg.png)  

Now we call *time_series_1d()* which extracts the relevant column and resamples it. Simply run the corresponding cell:  

![alt text](images/time_series_1d.png)  

Our raw data is ready. But before feeding the time-series predictor, we should split the data into train and test set; After that we can standardize and shingle each set.  
First set the relevant hyper-parameters:

![alt text](images/split_shingle.png)  

And now prepare the data for prediction by simply running the corresponding cell:

![alt text](images/prepare_data.png)

We can verify that our data is properly standardized and shingled:

![alt text](images/shingle_verify.png)
