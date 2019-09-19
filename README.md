# VPC Flow Logs Analytics
Analytics project provides a generic tool for analyzing Virtual Private Cloud (VPC) Flow Logs. 

## Why Use Analytics?
* It exploits VPC Flow Logs structure, gaining deep insights regarding traffic behavior in the cloud.
* It uses state-of-the-art machine learning techniques to find network anomalies.
* It displays the analytics in a comprehensive manner.

## Notebooks List
| Notebook | Action |
| -------- | ----------- |
| alerts | Alerts on massive traffic to a single destination | 
| anomalies_cnn | Detects anomalies in 1d time series using fully convolutional neural network |
| avg_packet | Plots a histogram of the average packet size distribution |
| daily_traffic | Plots a graph of the daily traffic in the network |
| hotspots | Detects network hotspots |
| in_out | Compares inbound to outbound traffic of an instance |
| port_distibution | Plots a chart of the distribution of the ports used in the network | 
| rejection_rate | Displays the rejection rate of an instance |
| tcp_flows | Compares one-way to two-way TCP connections |
| top_countries | Displays a table of the top talking countries |
| top_talking_pairs | Plots a graph of the top talking pairs |

## Prerequisites
* IPython Notebook with Python 3
* Flow Logs in [IBM format](data/format.md)

## Install Analytics Package
To install Analytics package and its dependencies, run the following command:  
`$ pip install git+https://github.com/cognetive/VPC-Flow-Logs-Analytics.git`

## Quick Start with IBM Watson Studio
### Import Analytics Notebook
To import Analytics notebook into a Watson Studio project, create a new notebook, select the *From URL* tab and provide the URL link for the notebook.
### Load Data
#### Option 1 - Load a Local File or a Data Asset
1. Open an empty code cell in your notebook.
2. Click the *Find and Add Data* icon, and then browse a data file or drag it into your notebook sidebar.
3. Click *Insert to code* link below the file and choose *Insert pandas DataFrame*.

#### Option 2 - Load from IBM Cloud Object Storage
To load data from a bucket in IBM COS, follow these steps:
1. Run `$ from data_preparation import download_file_cos`.
2. Go to IBM Cloud and choose *Service credentials* -> *New credential*. provide the reqired name and access role.
3. After the new credential was generated, click on the *View credentials* tab and copy its content.
4. Call *download_file_cos*(*credentials*, *bucket*, *file*). Provided with the credentials from step (3), this function loads *file* from *bucket* and returns it as a Pandas DataFrame.  

