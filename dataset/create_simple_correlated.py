import pandas as pd
import os
import random

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, './real_data_trace.csv') #5200 rows
temp_df = pd.read_csv(file_path)

switching_prob = 0.8

columns = ['channel0', 'channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6', 'channel7', 'channel8', 'channel9', 'channel10', 'channel11', 'channel12', 'channel13', 'channel14', 'channel15'] 

index = temp_df.index

df = pd.DataFrame(index=index, columns=columns)
df = df.fillna(0)

i = 0

on = True

for index, row in df.iterrows(): 
    
    if (on):
        df.loc[index, 'channel5'] = 1
    else:
        df.loc[index, 'channel13'] = 1

    if (random.random() > switching_prob):
        on = not on
    

df.to_csv('simple_correlated.csv')