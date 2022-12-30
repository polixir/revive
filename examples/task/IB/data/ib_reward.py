import torch
from typing import Dict


def get_reward(data : Dict[str, torch.Tensor]) -> torch.Tensor:
    obs = data["obs"]
    next_obs = data["next_obs"]

    single_reward = False
    if len(obs.shape) == 1:
        single_reward = True
        obs = obs.reshape(1, -1)
    if len(next_obs.shape) == 1:
        next_obs = next_obs.reshape(1, -1)

    CRF = 3.0
    CRC = 1.0

    fatigue = next_obs[:, 4]
    consumption = next_obs[:, 5]

    cost = CRF * fatigue + CRC * consumption

    reward = -cost

    if single_reward:
        reward = reward[0].item()
    else:
        reward = reward.reshape(-1, 1)

    return reward