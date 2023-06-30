import torch
from typing import Dict


def hidden_node_function(data : Dict[str, torch.Tensor]) -> torch.Tensor:
    return data["loss"]

def next_obs_node_function(data : Dict[str, torch.Tensor]) -> torch.Tensor:
    return data["obs"] + data["delta_obs"]

def predict_loss(data : Dict[str, torch.Tensor]) -> torch.Tensor:
    return data["loss"] 

def delta_obs_node_function(data : Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([data["delta_obs_1"], data["delta_obs_2"]], axis=-1)