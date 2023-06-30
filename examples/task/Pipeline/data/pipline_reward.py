import torch
from typing import Dict


def get_reward(data: Dict[str, torch.Tensor]) -> torch.Tensor:
    target = data["target"]
    next_obs = data["next_obs"][..., -1:]
    action = data["action"]

    reward = ((200 - torch.abs(next_obs - target)) * 0.01) ** 2
    reward -= (action ** 2) * 0.01
    return reward
