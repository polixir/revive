import ray
import torch
import pickle
import numpy as np
from copy import deepcopy
from torch.functional import F
from numbers import Number
from tianshou.data import Batch

import neorl

def soft_clamp(x : torch.Tensor, _min=None, _max=None):
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x

def get_input_from_names(batch : Batch, names : list):
    input = []
    for name in names:
        input.append(batch[name])
    return torch.cat(input, dim=-1)

def get_input_from_graph(graph : dict, 
                         output_name : str, 
                         batch_data : Batch):
    input_names = graph[output_name]
    inputs = []
    for input_name in input_names:
        inputs.append(batch_data[input_name])
    return torch.cat(inputs, dim=-1)

def get_sample_function(deterministic : bool):
    if deterministic:
        sample_fn = lambda dist: dist.mode
    else:
        sample_fn = lambda dist: dist.rsample()
    return sample_fn
    
def to_numpy(x):
    """Return an object without torch.Tensor."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    return x

def to_torch(x, dtype=torch.float32, device="cpu"):
    """Return an object without torch.Tensor."""

    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        x = x.to(device)
    elif isinstance(x, (np.number, np.bool_, Number)):
        x = to_torch(np.asanyarray(x), dtype, device)
    else:  # fallback
        x = np.asanyarray(x)
        if issubclass(x.dtype.type, (np.bool_, np.number)):
            x = torch.from_numpy(x).to(device)
            if dtype is not None:
                x = x.type(dtype)
        else:
            raise TypeError(f"object {x} cannot be converted to torch.")
    return x

def create_env(task : str):
    try:
        if task in ["HalfCheetah-v3", "Hopper-v3", "Walker2d-v3", "ib", "finance", "citylearn"]:
            env = neorl.make(task)
        elif task in ['halfcheetah-meidum-v0', 'hopper-medium-v0', 'walker2d-medium-v0']:
            import d4rl
            env = gym.make(task)
        else:
            env = gym.make(task)
    except:
        print(f'Warning: task {task} can not be created!')
        env = None

    return env