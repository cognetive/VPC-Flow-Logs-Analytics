# VPC Flow Logs Analytics
Analytics project provides a generic tool for analyzing Virtual Private Cloud (VPC) Flow Logs. 

## Why Use Analytics?
* It exploits VPC Flow Logs structure, gaining deep insights regarding traffic behavior in the cloud.
* It uses state-of-the-art machine learning techniques to find network anomalies.
* It displays the analytics in a comprehensive manner.

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

## Integreation with IBM Watson Studio
To import Analytics notebook into a Watson Studio project, create a new notebook, select the From URL tab and provide the URL link for the notebook.
