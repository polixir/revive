import torch
from typing import Dict


def get_next_step_node(data : Dict[str, torch.Tensor]) -> torch.Tensor:
    return data["step_node_"] + 1