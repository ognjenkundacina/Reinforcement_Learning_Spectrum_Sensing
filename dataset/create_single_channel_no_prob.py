import pandas as pd
import os

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, './real_data_trace.csv') #5200 rows
temp_df = pd.read_csv(file_path)


columns = ['channel0', 'channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6', 'channel7', 'channel8', 'channel9', 'channel10', 'channel11', 'channel12', 'channel13', 'channel14', 'channel15'] 

index = temp_df.index

df = pd.DataFrame(index=index, columns=columns)
df = df.fillna(0)

i = 0

for index, row in df.iterrows(): 
    
    if (i%16 == 0):
        df.loc[index, 'channel0'] = 1
    elif (i%16 == 1):
        df.loc[index, 'channel1'] = 1
    elif (i%16 == 2):
        df.loc[index, 'channel2'] = 1
    elif (i%16 == 3):
        df.loc[index, 'channel3'] = 1
    elif (i%16 == 4):
        df.loc[index, 'channel4'] = 1
    elif (i%16 == 5):
        df.loc[index, 'channel5'] = 1
    elif (i%16 == 6):
        df.loc[index, 'channel6'] = 1
    elif (i%16 == 7):
        df.loc[index, 'channel7'] = 1
    elif (i%16 == 8):
        df.loc[index, 'channel8'] = 1
    elif (i%16 == 9):
        df.loc[index, 'channel9'] = 1
    elif (i%16 == 10):
        df.loc[index, 'channel10'] = 1
    elif (i%16 == 11):
        df.loc[index, 'channel11'] = 1
    elif (i%16 == 12):
        df.loc[index, 'channel12'] = 1
    elif (i%16 == 13):
        df.loc[index, 'channel13'] = 1
    elif (i%16 == 14):
        df.loc[index, 'channel14'] = 1
    elif (i%16 == 15):
        df.loc[index, 'channel15'] = 1
    i += 1
    

df.to_csv('single_channel_no_prob.csv')