import gym
from gym import spaces
import random
import numpy as np 
from gym.spaces import Tuple
from gym.spaces import Space #todo provjeri treba li zadnje dvoje

#environment should have the entire dataset as an input parameter, but train and test methods
class Environment(gym.Env):

    def __init__(self):
        super(Environment, self).__init__()
        
        self.state = None
        self.state_space_dims = 256
        #self.action_space_dims = 1
        #actions are 0..15
        self.n_actions = 16

    def step(self, action, obs):
        #update state
        start_size = len(self.state)
        self.state += obs
        self.state = self.state[16:]
        next_state = self.state
        if (start_size != len(self.state)):
            print("Error in update state")

        reward = self.calculate_reward(action, obs)

        return next_state, reward

    #action takes values 0..15, so do indices of obs that has 16 values
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

