#drl vs mioptic policy vs wittle za perfectly correlated
# prije svega istreniraj drl za perfectly correlated
import os
import pandas as pd
from environment import Environment
import pandas as pd
from algorithms.deep_q_learning import DeepQLearningAgent
from common import get_initial_state_variables
import time

def load_dataset():
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, './dataset/perfectly_correlated.csv') #5200 rows
    #file_path = os.path.join(script_dir, './dataset/real_data_trace.csv') #5200 rows
    df = pd.read_csv(file_path)

    df_train = df[df.index <= 4479]
    df_test = df[df.index > 4479]

    return df_train, df_test

def main():
    #load dataset, divide into train and test
    df_train, df_test = load_dataset()

    #environment should have the entire dataset as an input parameter, but train and test methods
    environment = Environment()
    agent = DeepQLearningAgent(environment)

    n_episodes = 10
    print('agent training started')
    t1 = time.time()
    agent.train(df_train, n_episodes)
    t2 = time.time()
    print ('agent training finished in', t2-t1)

    agent.test(df_test)


if __name__ == '__main__':
  main()
