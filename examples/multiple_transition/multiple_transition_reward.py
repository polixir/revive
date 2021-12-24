import torch
from typing import Dict

def get_reward(data : Dict[str, torch.Tensor]) -> torch.Tensor:
    return data['o'][..., 1]