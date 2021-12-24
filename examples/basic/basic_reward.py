import torch
from typing import Dict

def get_reward(data : Dict[str, torch.Tensor]) -> torch.Tensor:
    return data['next_obs'][..., 0] - data['obs'][..., 0]