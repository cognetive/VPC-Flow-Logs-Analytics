#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd, numpy as np, ibm_boto3
from scipy import signal
from ibm_botocore.client import Config


# Loads file from IBM COS bucket and returns it as a Pandas Dataframe.
# Args: credentials - the service credentials. bucket - The bucket where the data is stored. file - The file to be loaded.
# Retruns: Dataframe.
def download_file_cos(credentials, bucket, file):  
    cos = ibm_boto3.client('s3',
                         ibm_api_key_id=cos_credentials['apikey'],
                         ibm_service_instance_id=cos_credentials['resource_instance_id'],
                         ibm_auth_endpoint='https://iam.bluemix.net/oidc/token',
                         config=Config(signature_version='oauth'),
                         endpoint_url='https://s3-api.us-geo.objectstorage.softlayer.net')
    cos.download_file(Bucket=bucket, Key=file, Filename='tmp.csv')
    return pd.read_csv('tmp.csv')

# Converts {Start, Last} columns to datetime type
# Args: df - Flowlogs DataFrame.
# Returns: Formatted DataFrame.
def data_format(df):
    return df.astype({'Start': 'datetime64[ms]', 'Last': 'datetime64[ms]'})


# Extracts and resamples a flowlogs attribute.
# Args: df - Flowlogs DataFrame. column - The attribute (column name) to be extracted. sample_rate - The sample rate.
# Returns - A resampled dataframe consists of one column.
def time_series(df, columns, sample_rate):
    return df.set_index('Start')[columns].resample(sample_rate).sum().fillna(0)


# Filters according to "Start" column and the given dates.
# Args: df - DataFrame in flow_logs format. (year1, month1, day1) - Start date. (year2, month2, day2) - end date.
# Retruns: A filtered DataFrame.
def date_filter(df, year1, month1, day1, year2, month2, day2):
    start_date = pd.to_datetime('%s-%s-%s' % (year1, month1, day1))
    end_date = pd.to_datetime('%s-%s-%s' % (year2, month2, day2))
    mask = (df['Start'] >= start_date) & (df['Start'] <= end_date)
    return df.loc[mask]


# Normalizes DataFrame columns to range [0, 1] using min-max method.
# Args: df - DataFrame.
# Returns: Normalized DataFrame.
def normalize(df):
    normalized_df = df.copy()
    for c in normalized_df.columns:
        normalized_df[c] = ((df[c] - np.min(df[c])) / (np.max(df[c]) - np.min(df[c]))).fillna(0)
    return normalized_df


# Standardizes DataFrame in a column-wise manner.
# Args: df - DataFrame.
# Returns: Standardized DataFrame.
def standardize(df):
    standardized_df = df.copy()
    for c in standardized_df.columns:
        standardized_df[c] = ((df[c] - np.mean(df[c])) / (np.std(df[c]))).fillna(0)
    return standardized_df


# Smoothes DataFrame in a column-wise manner using moving average.
# Args: df - DataFrame. w - Window size. s - Stride size.  
# Retruns: Smoothed DataFrame.
def smooth_MA(df, w, s=1):
    samples_num = int(len(df.index) / s) - w
    indices = [df.index[i*s+int(0.5*w)] for i in range(samples_num)]
    smoothed_df = pd.DataFrame(index=indices, columns=df.columns)
    for c in df.columns:
        smoothed_df[c] = [np.mean(df[c].values[i:i+w]) for i in samples_num]
    return smoothed_df


# Smoothes DataFrame in a column-wise manner using low-pass filter.
# Args: df - DataFrame. coeff - Amount of filter coefficients. thresh - The attenuation frequency. s - Stride size. 
# Retruns: Smoothed DataFrame.
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

