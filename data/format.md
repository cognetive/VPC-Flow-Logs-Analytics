## Data Format
Analytics uses Pandas and assumes IBM format for VPC Flow Logs. Thus, the data should be provided as a Pandas Dataframe named *flowlogs_df*, which contains the following fields (as columns):
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
