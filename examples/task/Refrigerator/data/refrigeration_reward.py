import torch
from typing import Dict


def get_reward(data : Dict[str, torch.Tensor]) -> torch.Tensor:
    '''
        data: contains all the node defined in the computational graph.
        return reward
    '''
    target_temperature = -2
    reward_temperature_weight = 1.0
    reward_power_weight = 1 - reward_temperature_weight

    reward_temperature = - torch.abs(data['next_temperature'][...,0:1] - target_temperature)
    reward_power = - torch.abs(data['action'])
    reward = (reward_temperature_weight * reward_temperature) + (reward_power_weight * reward_power)

    return reward