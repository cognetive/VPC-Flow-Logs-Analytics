## Anomaly Detection 
This tutorial demonstrates how to use Analytics to find anomalies in VPC Flow Logs. 
If you are still not familiar with Analytics, consider starting with the *hello world* tutorial.

### Preparing the Environment
1. Load *anomalies_cnn* into an IPython notebook running with Python 3.
2. Install Analytics (if needed).
3. Load the data into a Pandas DataFrame named *flowlogs_df*.

### Let's Start
First we select *ABBytes* as the metric in which we look for anomalies, and set the aggregation interval to be of 1-minute size:  
![alt text](images/metric_agg.png)
