import torch
from typing import Dict

def get_reward(data : Dict[str, torch.Tensor]) -> torch.Tensor:
    return data['obs'].sum(axis=-1)