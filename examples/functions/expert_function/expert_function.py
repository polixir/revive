import torch
from typing import Dict

def dynamics(inputs : Dict[str, torch.Tensor]) -> torch.Tensor:
    return inputs['obs']