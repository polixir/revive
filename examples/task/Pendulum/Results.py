import warnings
warnings.filterwarnings('ignore')

import gym
from IPython import display
import matplotlib
import matplotlib.pyplot as plt
import pickle
import copy
import numpy as np
import random

class get_results:
    def __init__(self,path_revive,path_old):
        self.policy_revive = pickle.load(open(path_revive, 'rb'))
        self.policy_old    = pickle.load(open(path_old, 'rb'))   
        self.env = gym.make('Pendulum-v1')
        self.action_bound = self.env.action_space.high[0]

    def take_revive_action(self, state):
        new_state = {}
        new_state['states'] = state
        action = self.policy_revive.infer(new_state)
        return action

    def roll_out(self, agent_num, step=200):

        return_revive_policy_on_true_env = []
        return_old_policy_on_true_env    = []

        ims_revive = []
        ims_old = []


        for agent in range(agent_num):

            epoch_return = 0
            state= self.env.reset(seed = agent)
            temp = []
            r_temp = []
            for i_step in range(step):
                temp.append(self.env.render(mode='rgb_array'))
                action = self.take_revive_action(state)
                next_state, reward, _, _ = self.env.step(action)
                r_temp.append(reward)
                epoch_return += reward
                state = next_state
            return_revive_policy_on_true_env.append(epoch_return)
            if epoch_return<=np.min(return_revive_policy_on_true_env):
                i_agent = agent
                self.ims_revive = temp
                self.r_revive = r_temp

            epoch_return = 0
            state= self.env.reset(seed = agent)
            temp = []
            r_temp = []
            for i_step in range(step):
                temp.append(self.env.render(mode='rgb_array'))
                action = self.policy_old.take_action(state)
                action = action.reshape(1,)
                next_state, reward, _, _ = self.env.step(action)
                r_temp.append(reward)
                epoch_return += reward
                state = next_state.reshape(3,)
            return_old_policy_on_true_env.append(epoch_return)
            if agent==i_agent:
                self.ims_old = temp 
                self.r_old = r_temp
        print('mean return of REVIVE: %.2f'%\
                np.mean(return_revive_policy_on_true_env))
        print('mean return of    old: %.2f'%\
                np.mean(return_old_policy_on_true_env))
        return np.mean(return_revive_policy_on_true_env), \
                np.mean(return_old_policy_on_true_env), \
                [self.ims_revive, self.r_revive], \
                [self.ims_old, self.r_old]                                       
        