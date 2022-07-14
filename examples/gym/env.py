import pickle

from copy import deepcopy
from gym import spaces
from gym.utils import seeding
from revive.utils.common_utils import get_reward_fn


class ReviveEnv:
    def __init__(self, 
                 env_model: str, 
                 action_node: str, 
                 reward_file_path:str=None, 
                 config_file:str=None):
        self.env = pickle.load(open(env_model, 'rb'), encoding='utf-8')
        self.graph = self.env.graph
        self.action_node = action_node

        self.init_space()
        self.init_reward_func(reward_file_path, config_file)

    def init_reward_func(self, reward_file_path, config_file):
        if (not reward_file_path) or (not config_file):
            self.reward_func = None
        else:
            self.reward_func = get_reward_fn(reward_file_path, config_file)

    def init_space(self):
        nodes = list(self.graph.keys())
        assert self.action_node in nodes, f"The '{self.action_node}' node isn't a learnable node. Please select action node from {nodes}."

        input_dim = 0
        for node_name in self.graph[self.action_node]:
            description = self.graph.descriptions[node_name]
            input_dim += len(description)
        action_dim = len(self.graph.descriptions[self.action_node])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (input_dim,))
        self.action_space = spaces.Box(low = -1, high = 1,shape = (action_dim,)) 

    def set_state(self, state:dict):
        self.state = state

    def reset(self, init_state:dict = None):
        if not init_state:
            raise NotImplementedError
        else:
            for node_name in self.graph[self.action_node]:
                description = self.graph.descriptions[node_name]
                assert len(description) == init_state[node_name].shape[1], f"Please check the init_state shape. It should be [B, {len(description)}]."
            self.set_state(init_state)

    def preprocess(self, actions):
        return self.env._env._data_postprocess(actions, self.action_node)

    def postprocess(self, actions):
        return self.env._env._data_postprocess(actions, self.action_node)

    def step(self, actions, **kwargs):
        actions = self.postprocess(actions)
        return self.infer(actions, **kwargs)

    def infer(self, actions, **kwargs):
        # get action inputs
        if kwargs:
            self.state = self.state.update(kwargs)
        pre_infer_result = self.env.infer_one_step(self.state)
        state = deepcopy(self.state)
        for node_name in self.graph.keys():
            if node_name == self.action_node:
                continue
            else:
                state[node_name] = pre_infer_result[node_name]
        # use action in env 
        state[self.action_node] = actions
        infer_result = self.env.infer_one_step(state)

        # get next step state
        self.state = self.graph.state_transition(infer_result)
        # get reward
        if self.reward_func:
            reward = self.reward_func(infer_result)
        else:
            reward = None

        return self.state, reward, False, {}

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

if __name__ == "__main__":
    import numpy as np
    from revive.utils.common_utils import load_data
    
    env = ReviveEnv("./env.pkl","act", "../../data/test_reward.py", "../../data/test.yaml")
    data = load_data("../../data/test.npz")
    state = {"obs":data["obs"][:10,:]}
    env.reset(state)
    print(env.infer(data["act"][:10,:]))