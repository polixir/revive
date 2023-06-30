import torch
from torch import nn
import torch.nn.functional as F
import random
import gym
import numpy as np
import collections


class Expert_net(torch.nn.Module):
    # 这是一个简单的两层神经网络
    def __init__(self, num_in, num_out, hidden_dim, bound):
        super().__init__()
        self.fc1 = nn.Linear(num_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_out)

        self.activation = F.relu
        # self.out_fn = out_fn
        self.bound = bound

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        # x = torch.sigmoid(self.fc3(x))
        return x*self.bound


#进行策略加载
class Expert_policy:
    def __init__(self):
        self.env_name = 'Pendulum-v1'
        self.env = gym.make(self.env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.hidden_dim = 64
        self.action_bound = self.env.action_space.high[0]
        self.limit = 0.5

    def enport_net(self, PATH):
        self.Load_policy = Expert_net(self.state_dim, self.action_dim, 
                                      self.hidden_dim, self.action_bound)
        # PATH = "Expert_pendulum_v1.pt"
        self.Load_policy.load_state_dict(torch.load(PATH))
        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in self.Load_policy.state_dict():
            print(param_tensor, "\t", self.Load_policy.state_dict()[param_tensor].size())

        self.Load_policy = self.Load_policy.eval()

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float)
        if np.random.random() < self.limit:
            action = self.Load_policy(state).detach().numpy()
        else:
            action = np.array([(2*np.random.random()-1)*self.action_bound], dtype=np.float32)
        # action = action + self.sigma * np.random.randn(action_dim)
        return action

    def sampling(self,):
        random.seed(0)
        np.random.seed(0)
        # env.reset(seed = 0)
        expert_buffer = collections.deque()
        stop_index = []
        for i in range(100):
            self.env.seed(i)
            state = self.env.reset()
            done = False
            while not done:
                action = self.take_action(state).reshape(-1)
                next_s, _, done, _ = self.env.step(action)
                expert_buffer.append((state, action, next_s))
                state = next_s
            stop_index.append(len(expert_buffer))
        return expert_buffer, stop_index