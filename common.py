import pandas as pd

def get_initial_state_variables(df):
    i=0    
    row_list = []
    for index, rows in df.iterrows(): 
        # Create list for the current row 
        temp_list =[rows.channel0, rows.channel1, rows.channel2, rows.channel3, rows.channel4, rows.channel5, rows.channel6, rows.channel7, rows.channel8, rows.channel9, rows.channel10, rows.channel11, rows.channel12, rows.channel13, rows.channel14, rows.channel15] 
        
        # append the list to the final list 
        row_list += temp_list 
        i += 1
        if (i == 16):
            break

    if (len(row_list) != 256):
        print ("Error: state variable list should have 256 members")
    return row_list

def get_obs_from_df_row(row):
    temp_list =[row.channel0, row.channel1, row.channel2, row.channel3, row.channel4, row.channel5, row.channel6, row.channel7, row.channel8, row.channel9, row.channel10, row.channel11, row.channel12, row.channel13, row.channel14, row.channel15] 
    return temp_list

def split_dataframe(df):
    step = 200
    num_splits = len(df) // step
    df_list = []
    lower_index = 0

    for i in range (num_splits):
        df_list.append(df[ (df.index >= lower_index) * ((df.index < lower_index + step)) ])
        lower_index += step

    if (len(df) != len(df_list[1]) * num_splits ):
        print ("Warning: trainset may not be split correctly")

    return df_list