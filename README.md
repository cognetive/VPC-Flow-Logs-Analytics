# VPC Flow Logs Analytics
Analytics project provides a generic tool for analyzing Virtual Private Cloud (VPC) Flow Logs. 

## Why Use Analytics?
* It exploits VPC Flow Logs structure, gaining deep insights regarding traffic behavior in the cloud.
* It uses state-of-the-art machine learning techniques to find network anomalies.
* It displays the analytics in a comprehensive manner.

## Install Analytics Package
To install Analytics package and its dependencies, run the following command:  
`$ pip install git+https://github.com/cognetive/VPC-Flow-Logs-Analytics.git`

## Data Format
Analytics uses Pandas and assumes IBM format for VPC Flow Logs. Thus, the data should be provided as a Pandas Dataframe which contains the following fields (as columns):
- Start
- Last
- Status
- Action
- Protocol
- src_ip
- dst_ip
- src_port
- dst_port
- ABPackets
- BAPackets
- ABBytes
- BABytes

## Quick Start with IBM Watson Studio
### Import Analytics Notebook
To import Analytics notebook into a Watson Studio project, create a new notebook, select the *From URL* tab and provide the URL link for the notebook.
### Load Data
#### Option 1 - Local File or Data Asset
To load data from a local file or from a data asset in Watson Studio: 
1. Open an empty code cell in your notebook.
2. Click the *Find and Add Data* icon, and then browse a data file or drag it into your notebook sidebar.
3. Click *Insert to code* link below the file and choose *Insert pandas DataFrame*.
Now, as the data is loaded into the notebook, you can start analyzing it! 
Note: You might find *data_preparation* module useful for preprocessing your data before analyzing it. See *data_preparation* section under the documentation for further reading. 
