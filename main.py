import os
from environment import Environment
import pandas as pd
from algorithms.deep_q_learning import DeepQLearningAgent
from common import get_initial_state_variables
import time

def load_dataset():
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, './dataset/real_data_trace.csv') #5200 rows
    df = pd.read_csv(file_path)
    least_good_columns = ['channel0', 'channel1', 'channel2', 'channel3', 'channel5', 'channel6', 'channel7', 'channel11'] 
    df = df[least_good_columns]
    df_train = df[df.index <= 4599]
    df_test = df[df.index > 4599]

    return df_train, df_test

def main():
    #load dataset, divide into train and test
    df_train, df_test = load_dataset()

    #environment should have the entire dataset as an input parameter, but train and test methods
    environment = Environment()
    agent = DeepQLearningAgent(environment)

    n_episodes = 800
    ####n_episodes = 25
    episode_length = 200
    print('agent training started')
    t1 = time.time()
    agent.train(df_train, n_episodes, episode_length)
    t2 = time.time()
    print ('agent training finished in', t2-t1)

    print ('Test on the test dataset')
    agent.test(df_test)
    print ('Test on the train dataset')
    agent.test(df_train)


if __name__ == '__main__':
  main()
