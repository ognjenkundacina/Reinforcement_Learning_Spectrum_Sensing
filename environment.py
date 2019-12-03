import gym
from gym import spaces
import random
import numpy as np 

#environment should have the entire dataset as an input parameter, but train and test methods
class Environment(gym.Env):

    def __init__(self):
        super(Environment, self).__init__()
        
        self.state = None
        self.state_space_dims = 256
        #self.action_space_dims = 1
        #actions are 0..7
        self.n_actions = 8
        
    def _obs_intersect_action(self, obs, action):
        temp = [0 for j in range(self.n_actions)]
        if (obs[action] == 1):
            temp[action] = 1
        elif (obs[action] == 0):
            temp[action] = -1
        #print ('TEMP: ', temp)
        return temp

    def step(self, action, obs):
        #update state
        start_size = len(self.state)
        self.state += self._obs_intersect_action(obs, action)
        self.state = self.state[self.n_actions:]
        next_state = self.state
        if (start_size != len(self.state)):
            print("Error in update state")

        if (256 != len(self.state)):
            print("Error: state size not equal to 256")

        reward = self.calculate_reward(action, obs)

        return next_state, reward

    #action takes values 0..7, so do indices of obs that has 8 values
    def calculate_reward(self, action, obs):
        reward = 0.0
        if (obs[action] == 1):
            reward = 1.0
        elif (obs[action] == 0):
            reward = -1.0
        else:
            print ("Error: channel quality should be 1 or 0")
        return reward

    def reset(self, state_variables):
        self.state = state_variables
        return self.state

