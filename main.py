import os
from environment import Environment
import pandas as pd
from algorithms.deep_q_learning import DeepQLearningAgent
from common import get_initial_state_variables
import time
import matplotlib.pyplot as plt 

def load_dataset(split_index):
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, './dataset/simple_correlated.csv') #5200 rows
    #file_path = os.path.join(script_dir, './dataset/perfectly_correlated.csv') #5500 rows
    #file_path = os.path.join(script_dir, './dataset/perfectly_correlated5200.csv') #5200 rows
    #file_path = os.path.join(script_dir, './dataset/real_data_trace.csv') #5500 rows
    #file_path = os.path.join(script_dir, './dataset/single_channel_no_prob.csv') #5200 rows
    #file_path = os.path.join(script_dir, './dataset/single_channel_no_prob_strange_order.csv') #5200 rows
    #file_path = os.path.join(script_dir, './dataset/single_channel_with_switching_prob.csv') #5200 rows
    #file_path = os.path.join(script_dir, './dataset/multiple_channels_with_switching_prob.csv') #5200 rows
    #file_path = os.path.join(script_dir, './dataset/multiple_channels_no_prob.csv') #5200 rows

    df = pd.read_csv(file_path)

    df_train = df[df.index < split_index]
    df_test = df[df.index >= split_index]

    return df_train, df_test

def main():

    split_index = 4600
    df_train, df_test = load_dataset(split_index)

    environment = Environment()
    agent = DeepQLearningAgent(environment)

    n_episodes = 800
    episode_length = 200
    print('agent training started')
    t1 = time.time()
    agent.train(df_train, n_episodes, episode_length)
    t2 = time.time()
    print ('agent training finished in', t2-t1)

    print ('Test on the test dataset')
    transfered_messages_percent = agent.test(df_test)

if __name__ == '__main__':
  main()
