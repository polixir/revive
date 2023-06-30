import torch
import numpy as np
from typing import Dict


def get_reward(data : Dict[str, torch.Tensor]) -> torch.Tensor:
    action = data["action"]
    delta_x = data["delta_x"]
    
    
    forward_reward_weight = 1.0 
    ctrl_cost_weight = 0.1
    dt = 0.05
    
    if isinstance(action, np.ndarray):
        array_type = np
        ctrl_cost = ctrl_cost_weight * array_type.sum(array_type.square(action),axis=-1, keepdims=True)
    else:
        array_type = torch
        ctrl_cost = ctrl_cost_weight * array_type.sum(array_type.square(action),axis=-1, keepdim=True)
    
    x_velocity = delta_x / dt
    forward_reward = forward_reward_weight * x_velocity
    
    reward = forward_reward - ctrl_cost
    
    return reward