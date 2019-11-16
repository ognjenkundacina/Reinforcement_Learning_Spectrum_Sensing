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
    print('agent training started')
    t1 = time.time()
    agent.train(df_train, n_episodes)
    t2 = time.time()
    print ('agent training finished in', t2-t1)

    print ('Test on the test dataset')
    agent.test(df_test)
    print ('Test on the train dataset')
    agent.test(df_train)

    #TODO
    #1. Implementiraj test metodu koja sabira reward, ili mozda discounted reward DONE
    #1a Smanji target update na 2, pokusaj i sa 3, 4 DONE
    #1c Nemoj citav trening set za epizodu, razbij ga na manje dijelove, povecaj broj epizoda, i vrati target update na 10 onda DONE

    #PROBAJ SINGLE CHANNEL DATASET DA NAPRAVIS!!!!!!!!!!!!!!!!!!!!!!!
    #Procitaj rad detaljnije, parametre, arhitekturu mreze
    #1d Idi korak po korak po implelemtacji, vidi koji su ulazi u mrezu tokom testiranja, kakve izlaze daje
    #1e Uporedi sa implementacijom koju si vec nasao!

    #2. implementiraj whittle index, posto sa njim najvise poredee 
    #3. Real data trace testiraj na 8 kanala, a ne na 16
    #4. Razmisli malo o dizajnu, na previse mjesta je hardkodovano 256, sta cemo kad je 8 kanala, onda je i obs drugaciji i get action = novi  #projekat
    #5. Na real datatrace vidi da li je bolje staviti 32 zadnja stanja u state umjesto 16... ako jeste onda ima smisla pokusavati rekurentno!
    #6. Da li bi akcija trebalo da utice na sljedece stanje ili samo da citamo iz dataseta?


if __name__ == '__main__':
  main()
