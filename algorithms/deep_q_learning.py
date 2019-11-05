from collections import namedtuple
from itertools import count
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from common import get_initial_state_variables, get_obs_from_df_row

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, output_size)
        #self.fc2_bn = nn.BatchNorm1d(200) consider this

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DeepQLearningAgent:

    def __init__(self, environment):
        self.environment = environment
        self.epsilon = 0.1
        self.gamma = 0.99 #best, tested
        self.batch_size = 32
        self.target_update = 10
        self.memory = ReplayMemory(1000000)

        self.state_space_dims = environment.state_space_dims
        self.n_actions = environment.n_actions

        self.policy_net = DQN(self.state_space_dims, self.n_actions)
        self.target_net = DQN(self.state_space_dims, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.policy_net.train() #train mode (train vs eval mode)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001) 

    #returns channel id: 0..15
    def get_action(self, state):
        if random.random() > self.epsilon:
            self.policy_net.eval()
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action_index = self.policy_net(state).max(1)[1].view(1, 1)
                self.policy_net.train()
                return action_index #todo check if it is 0..15
        else:
            return torch.tensor([[random.randint(0, self.n_actions-1)]], dtype=torch.int)    

    def reset_environment_training(self, df_train):
        train_initial_state_variables = get_initial_state_variables(df_train)
        return self.environment.reset(train_initial_state_variables)

    def train(self, df_train, n_episodes):

        for i_episode in range(n_episodes):
            if (i_episode % 1 == 0):
                print ("Episode: ", i_episode)

            state = self.reset_environment_training(df_train) 
            state = torch.tensor([state], dtype=torch.float)
            total_episode_reward = 0 
            total_episode_reward = torch.tensor([total_episode_reward], dtype=torch.float).view(-1,1)
            i = 0

            for index, rows in df_train.iterrows(): 
                i += 1
                if (i <= 16):
                    continue
                action = self.get_action(state) #0..15
                if (action > 15):  #todo ovo skloni kasnije zbog brzine
                    print ("Error, action has value greater than 15")
                obs = get_obs_from_df_row(rows)
                next_state, reward = self.environment.step(action, obs)
                reward = torch.tensor([reward], dtype=torch.float)
                action = torch.tensor([action], dtype=torch.int)
                next_state = torch.tensor([next_state], dtype=torch.float)
                if (index == df_train.index[-1]):
                    next_state = None

                self.memory.push(state, action, next_state, reward)
                                
                total_episode_reward += reward
                state = next_state

                self.optimize_model()

            if (i_episode % 1 == 0):
                print ("total_episode_reward: ", total_episode_reward)

            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        torch.save(self.policy_net.state_dict(), "policy_net")   

    def test(self, df_test):
        test_initial_state_variables = get_initial_state_variables(df_test)
        state = self.environment.reset(test_initial_state_variables) #tddo use reset_environment_training



    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)

        #converts batch array of transitions to transiton of batch arrays
        batch = Transition(*zip(*transitions))

        #compute a mask of non final states and concatenate the batch elements
        #there will be zero q values for final states later... therefore we need mask
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype = torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).view(-1,1)
        reward_batch = torch.cat(batch.reward).view(-1,1)

        # compute Q(s_t, a) - the model computes Q(s_t), then we select
        # the columns of actions taken. These are the actions which would've
        # been taken for each batch state according to policy net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.long())

        #if (i_episode%1000 == 0 and episode_iterator == 1):
        #    g = open(str(i_episode)+"policy_net.txt", 'w+')
        #    g.write(str(self.policy_net(state_batch).tolist()))

        #gather radi isto sto i:
        #q_vals = []
        #for qv, ac in zip(Q(obs_batch), act_batch):
        #    q_vals.append(qv[ac])
        #q_vals = torch.cat(q_vals, dim=0)

        # Compute V(s_{t+1}) for all next states
        # q values of actions for non terminal states are computed using
        # the older target network, selecting the best reward with max
        next_state_values = torch.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach() #manje od 128 stanja, nema final stanja
        #za stanja koja su final ce next_state_values biti 0
        #detach znaci da se nad varijablom next_state_values ne vrsi optimizacicja
        next_state_values = next_state_values.view(-1,1)
        # compute the expected Q values
        expected_state_action_values = (next_state_values*self.gamma) + reward_batch

        #Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()