import pandas as pd
import os
import random

def get_obs_from_df_row(row):
    temp_list =[row.channel0, row.channel1, row.channel2, row.channel3, row.channel4, row.channel5, row.channel6, row.channel7, row.channel8, row.channel9, row.channel10, row.channel11, row.channel12, row.channel13, row.channel14, row.channel15] 
    return temp_list

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, './real_data_trace.csv') #5200 rows
df = pd.read_csv(file_path)

columns = ['channel0', 'channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6', 'channel7', 'channel8', 'channel9', 'channel10', 'channel11', 'channel12', 'channel13', 'channel14', 'channel15'] 

channelQualityList = [0 for i in range(16)]

for index, row in df.iterrows(): 
    obs = get_obs_from_df_row(row)
    for i in range(16):
        if (obs[i] == 1):
            channelQualityList[i] += 1
    
for i in range(16):
    print(i, channelQualityList[i])


"""
Rezultati:
0 240
1 6
2 1635
3 1427
4 2787  best
5 153
6 9
7 1501
8 3883  best
9 4506  best
10 2623 best
11 2020
12 2513 best
13 2174 best
14 3647 best
15 3772 best

least good channels:
columns = ['channel0', 'channel1', 'channel2', 'channel3', 'channel5', 'channel6', 'channel7', 'channel11'] 


"""