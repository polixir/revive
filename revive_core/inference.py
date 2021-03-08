import torch
import numpy as np
from copy import deepcopy
from tianshou.data import Batch

from revive_core.utils import *

@ray.remote
def test_one_trail(env, policy):
    env = deepcopy(env)
    policy = deepcopy(policy)

    obs = env.reset()
    reward = 0
    length = 0
    while True:
        action = policy.infer({'obs' : obs[np.newaxis]})[0]
        obs, r, done, info = env.step(action)
        reward += r
        length += 1

        if done:
            break

    return (reward, length)

def test_on_real_env(env, policy, number_of_runs=10):
    rewards = []
    episode_lengths = []

    results = ray.get([test_one_trail.remote(env, policy) for _ in range(number_of_runs)])
    rewards = [result[0] for result in results]
    episode_lengths = [result[1] for result in results]

    return np.mean(rewards), np.mean(episode_lengths)

def load_policy(filename : str, policy_name : str=None):
    try:
        model = torch.load(filename, map_location='cpu')
    except:
        with open(filename, 'rb') as f:
            model = pickle.load(f)

    if isinstance(model, VirtualEnv):
        model = model._env

    if isinstance(model, VirtualEnvDev):
        i = find_policy_index(model.graph, policy_name)
        _model = model.models[i]
        model = PolicyModelDev(_model, model.graph, model.data_pre_fn, model.data_post_fn, policy_name)

    if isinstance(model, PolicyModelDev):
        model = PolicyModel(model)

    return model

def find_policy_index(graph : dict, policy_name : str):
    for i, k in enumerate(graph.keys()):
        if k == policy_name:
            break
    return i

class DataProcessor:
    """
        This class deal with the data mapping between raw inputs and the form handled by network
    """

    def __init__(self, data_configs, processing_params, orders):
        self.data_configs = data_configs
        self.processing_params = processing_params
        self.orders = orders

    @property
    def keys(self):
        return list(self.data_configs.keys())

    # ----------------------------------------------------------------------------------- #
    #                                Fuctions for Tensor                                  # 

    def _process_fn_torch(self, data, data_config, processing_params, order):
        data = data[..., order['forward']]
        processed_data = []
        for config, s, (mean, std) in zip(data_config, processing_params['forward_slices'], processing_params['norms']):
            _data = data[..., s]
            if config['type'] == 'category':
                onehot = torch.zeros(_data.shape[0], config['dim'], dtype=_data.dypte, device=_data.device)
                _data = torch.clamp_max(_data, config['dim'] - 1) # make sure data within the region
                _data = _data.view(-1).long()
                onehot[torch.arange(_data.shape[0]), _data] = 1
                processed_data.append(onehot)
            elif config['type'] == 'continuous':
                mean = torch.tensor(mean.copy(), dtype=_data.dtype, device=_data.device)
                std = torch.tensor(std.copy(), dtype=_data.dtype, device=_data.device)
                min_value = mean - std
                max_value = mean + std
                # _data = torch.clamp(_data, min_value, max_value)
                _data = (_data - mean) / std
                processed_data.append(_data)
            elif config['type'] == 'discrete':
                mean = torch.tensor(mean.copy(), dtype=_data.dtype, device=_data.device)
                std = torch.tensor(std.copy(), dtype=_data.dtype, device=_data.device)
                min_value = mean - std
                max_value = mean + std
                # _data = torch.clamp(_data, min_value, max_value)
                # dequantization
                # _data = _data + torch.rand_like(_data)       
                _data = (_data - mean) / std   
                processed_data.append(_data) 

        return torch.cat(processed_data, dim=-1)

    def _deprocess_fn_torch(self, data, data_config, processing_params, order):
        processed_data = []
        for config, s, (mean, std) in zip(data_config, processing_params['backward_slices'], processing_params['norms']):
            _data = data[..., s]
            if config['type'] == 'category':
                _data = torch.argmax(_data, axis=-1).float()
                _data = _data.unsqueeze(-1)
                processed_data.append(_data)
            elif config['type'] == 'continuous':
                mean = torch.tensor(mean.copy(), dtype=_data.dtype, device=_data.device)
                std = torch.tensor(std.copy(), dtype=_data.dtype, device=_data.device)
                _data = _data * std + mean
                processed_data.append(_data)
            elif config['type'] == 'discrete':
                mean = torch.tensor(mean.copy(), dtype=_data.dtype, device=_data.device)
                std = torch.tensor(std.copy(), dtype=_data.dtype, device=_data.device)
                _data = _data * std + mean      
                # _data = torch.floor(_data)  
                _data = torch.round(_data) # correct deprocess without dequantization
                processed_data.append(_data) 

        processed_data = torch.cat(processed_data, axis=-1)   
        processed_data = processed_data[..., order['backward']]
        return processed_data  

    def process_single_torch(self, data : torch.Tensor, key: str) -> torch.Tensor:
        """
        Preprocess single data according different types of data including 'category', 'continuous', and 'discrete'.

        """
        if key in self.keys:
            return self._process_fn_torch(data, self.data_configs[key], self.processing_params[key], self.orders[key])
        else: # do nothing
            return data

    def deprocess_single_torch(self, data : torch.Tensor, key: str) -> torch.Tensor:
        """
        Post process single data according different types of data including 'category', 'continuous', and'discrete'.

        """
        if key in self.keys:
            return self._deprocess_fn_torch(data, self.data_configs[key], self.processing_params[key], self.orders[key])
        else: # do nothing
            return data

    def process_torch(self, data):
        """
        Preprocess batch data according different types of data including 'category', 'continuous', and'discrete'.

        """
        return Batch({k : self.process_single_torch(data[k], k) for k in data.keys()})
    
    def deprocess_torch(self, data):
        """
        Post process batch data according different types of data including 'category', 'continuous', and'discrete'.

        """
        return Batch({k : self.deprocess_single_torch(data[k], k) for k in data.keys()})

    # ----------------------------------------------------------------------------------- #
    #                                Fuctions for ndarray                                 # 

    def _process_fn(self, data, data_config, processing_params, order):
        data = data.take(order['forward'], axis=-1)
        processed_data = []
        for config, s, (mean, std) in zip(data_config, processing_params['forward_slices'], processing_params['norms']):
            _data = data[..., s]
            if config['type'] == 'category':
                onehot = np.zeros((_data.shape[0], config['dim']), dtype=np.float32)
                _data = np.clip(_data, -float('inf'), config['dim'] - 1) # make sure data within the region
                _data = _data.reshape((-1)).astype(int)
                onehot[np.arange(_data.shape[0]), _data] = 1
                processed_data.append(onehot)
            elif config['type'] == 'continuous':
                min_value = mean - std
                max_value = mean + std
                # _data = np.clip(_data, min_value, max_value)
                _data = (_data - mean) / std
                processed_data.append(_data)
            elif config['type'] == 'discrete':
                min_value = mean - std
                max_value = mean + std
                # _data = np.clip(_data, min_value, max_value)
                # dequantization
                # _data = _data + np.random.rand(*_data.shape).astype(np.float32)       
                _data = (_data - mean) / std   
                processed_data.append(_data) 

        return np.concatenate(processed_data, axis=-1)

    def _deprocess_fn(self, data, data_config, processing_params, order):
        processed_data = []
        for config, s, (mean, std) in zip(data_config, processing_params['backward_slices'], processing_params['norms']):
            _data = data[..., s]
            if config['type'] == 'category':
                _data = np.argmax(_data, axis=-1).astype(np.float32)
                _data = _data.reshape([*_data.shape, 1])
                processed_data.append(_data)
            elif config['type'] == 'continuous':
                _data = _data * std + mean
                processed_data.append(_data)
            elif config['type'] == 'discrete':
                _data = _data * std + mean      
                # _data = np.floor(_data)  
                _data = np.round(_data) # correct deprocess without dequantization
                processed_data.append(_data) 

        processed_data = np.concatenate(processed_data, axis=-1)   
        processed_data = processed_data.take(order['backward'], axis=-1)
        return processed_data  

    def process_single(self, data : np.ndarray, key: str) -> np.ndarray:
        """
        Preprocess single data according different types of data including 'category', 'continuous', and'discrete'.

        """
        if key in self.keys:
            return self._process_fn(data, self.data_configs[key], self.processing_params[key], self.orders[key])
        else: # do nothing
            return data

    def deprocess_single(self, data : np.ndarray, key: str) -> np.ndarray:
        """
        Post process single data according different types of data including 'category', 'continuous', and'discrete'.

        """
        if key in self.keys:
            return self._deprocess_fn(data, self.data_configs[key], self.processing_params[key], self.orders[key])
        else: # do nothing
            return data

    def process(self, data):
        """
        Preprocess batch data according different types of data including 'category', 'continuous', and'discrete'.

        """
        return Batch({k : self.process_single(data[k], k) for k in data.keys()})
    
    def deprocess(self, data):
        """
        Post process batch data according different types of data including 'category', 'continuous', and'discrete'.

        """
        return Batch({k : self.deprocess_single(data[k], k) for k in data.keys()})

class VirtualEnvDev(torch.nn.Module):
    def __init__(self, models, graph, data_pre_fn, data_post_fn):
        super(VirtualEnvDev, self).__init__()
        self.models = torch.nn.ModuleList(models)
        self.graph = graph
        self.data_pre_fn = data_pre_fn
        self.data_post_fn = data_post_fn
        self.set_target_policy_name(list(self.graph.keys())[0]) # default

    def set_target_policy_name(self, target_policy_name):
        self.target_policy_name = target_policy_name

        # find target index
        for i, (output_name, input_names) in enumerate(self.graph.items()):
            if output_name == self.target_policy_name:
                self.index = i
                break

    def _data_preprocess(self, data, data_key="obs"):
        data = self.data_pre_fn(data, data_key)
        data = to_torch(data)

        return data

    def _data_postprocess(self, data, data_key):
        data = to_numpy(data)
        data = self.data_post_fn(data, data_key)
        
        return data

    def _infer_one_step(self, state, action=None, deterministic=True):
        sample_fn = get_sample_function(deterministic)

        for k in list(state.keys()):
            state[k] = self._data_preprocess(state[k], k)

        state = self.pre_computation(state, deterministic)

        if action is None:
            state = self.forward(state, deterministic)
        else:
            action = self._data_preprocess(action, self.target_policy_name)
            state[self.target_policy_name] = action

        state = self.post_computation(state, deterministic)

        for k in list(state.keys()):
            state[k] = self._data_postprocess(state[k], k)

        return state

    def infer_k_steps(self, init_state, k, deterministic=True):
        assert k > 0 and isinstance(k, int)

        outputs = []
        for i in range(k):
            output = self._infer_one_step(init_state, deterministic=deterministic)
            outputs.append(output)
            init_state = dict(obs=output['next_obs'])

        return outputs

    def infer_one_step(self, state, action, deterministic=True):
        return self._infer_one_step(state, action, deterministic=deterministic)

    def forward(self, batch : Batch, deterministic : bool = True):
        '''run the target node'''
        sample_fn = get_sample_function(deterministic)

        output_name, input_names = list(self.graph.items())[self.index]
        model = self.models[self.index]
        input = get_input_from_names(batch, input_names)
        dist = model(input)
        batch[output_name] = sample_fn(dist)

        return batch

    def pre_computation(self, batch : Batch, deterministic : bool = True):
        '''run all the node before target node. skip if the node value is already available.'''
        sample_fn = get_sample_function(deterministic)

        for (output_name, input_names), model in zip(list(self.graph.items())[:self.index], self.models[:self.index]):
            if not output_name in batch.keys():
                input = get_input_from_names(batch, input_names)
                dist = model(input)
                batch[output_name] = sample_fn(dist)
            else:
                print(f'Skip {output_name}, since it is provided in the inputs.')

        return batch

    def post_computation(self, batch : Batch, deterministic : bool = True):
        '''run all the node after target node'''
        sample_fn = get_sample_function(deterministic)

        for (output_name, input_names), model in zip(list(self.graph.items())[self.index+1:], self.models[self.index+1:]):
            input = get_input_from_names(batch, input_names)
            dist = model(input)
            batch[output_name] = sample_fn(dist)

        return batch

class VirtualEnv:
    def __init__(self, env_list):
        self._env = env_list[0]
        self.env_list = env_list
        self.graph = self._env.graph

    @property
    def target_policy_name(self):
        return self._env.target_policy_name

    def set_target_policy_name(self, target_policy_name):
        for env in self.env_list:
            env.set_target_policy_name(target_policy_name)

    @torch.no_grad()
    def infer_k_steps(self, init_state : dict, k : int=1, deterministic=True):
        """
        Generate k steps interactive data

        :param init_state: a dict of initial input nodes

        :param k: how many steps to generate

        :param deterministic: 
            if True, the most likely actions are generated; 
            if False, actions are generated by sample.
            Default: True

        :return: k steps interactive data

        """
        return self._env.infer_k_steps(init_state, k, deterministic)

    @torch.no_grad()
    def infer_one_step(self, state : dict, action : np.ndarray, deterministic=True):
        """
        Generate one step interactive data given action.

        :param state: a dict of input nodes

        :param action: action

        :param deterministic: 
            if True, the most likely actions are generated; 
            if False, actions are generated by sample.
            Default: True

        :return: one step outputs

        """        
        return self._env.infer_one_step(state, action, deterministic)

class PolicyModelDev:
    def __init__(self, model, graph, data_pre_fn, data_post_fn, target_policy_name):
        self.model = model
        self.graph = graph
        self.data_pre_fn = data_pre_fn
        self.data_post_fn = data_post_fn
        self.set_target_policy_name(target_policy_name)

    def set_target_policy_name(self, target_policy_name):
        self.target_policy_name = target_policy_name

        # find target index
        for i, (output_name, input_names) in enumerate(self.graph.items()):
            if output_name == self.target_policy_name:
                self.index = i
                break

    def _data_preprocess(self, data, data_key="obs"):
        data = self.data_pre_fn(data, data_key)
        data = to_torch(data)

        return data

    def _data_postprocess(self, data, data_key="action1"):
        data = to_numpy(data)
        data = self.data_post_fn(data, data_key)

        return data

    def infer(self, state : dict, deterministic=True):
        sample_fn = get_sample_function(deterministic)

        for k, v in state.items():
            state[k] = self._data_preprocess(v, data_key=k)

        state = get_input_from_graph(self.graph, self.target_policy_name, state)
        if 'offlinerl' in str(type(self.model)):
            action = self.model.get_action(state)
        else:
            action = sample_fn(self.model(state))

        action = self._data_postprocess(action, self.target_policy_name)

        return action

class PolicyModel:
    def __init__(self, policy_model_dev):
        self._policy_model = policy_model_dev
        self.graph = self._policy_model.graph

    @property
    def target_policy_name(self):
        return self._policy_model.target_policy_name

    @torch.no_grad()
    def infer(self, state : dict, deterministic=True):
        """
        Generate action according policy.

        :param state: a dict contain *ALL* the input nodes of the policy node

        :param deterministic: 
            if True, the most likely actions are generated; 
            if False, actions are generated by sample.
            Default: True

        :return: action

        """
        return self._policy_model.infer(state, deterministic)