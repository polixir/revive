import torch
from typing import Dict


def next_ts_placeholder_transition_function(data : Dict[str, torch.Tensor]) -> torch.Tensor:
    next_obs = data['next_'+'placeholder']
    dim = next_obs.shape[-1]
    ts_obs = data['ts_'+'placeholder']
    return torch.cat([ts_obs, next_obs],axis=-1)[..., dim:]