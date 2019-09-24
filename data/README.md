## Data Format
Analytics uses Pandas and assumes IBM format for VPC Flow Logs. Thus, the data should be provided as a Pandas DataFrame named *flowlogs_df*, which contains (at least) the following fields (as columns):  

| Field | Description |
| --- | --- |
| Start | The start time of the flow |
| Last | The end time of the flow |
| Action | Whether the flow was accepted or rejected |
| Protocol | The protocol number of the traffic |
| src_ip | The source IP address |
| dst_ip | The destination IP address |
| src_port | The source port of the traffic |
| dst_port| The destination port of the traffic |
| ABPackets | The number of packets from source to destination transferred during the flow |
| BAPackets | The number of packets from destination to source transferred during the flow |
| ABBytes | The number of bytes from source to destination transferred during the flow |
| BABytes | The number of bytes from destination to source transferred during the flow |
