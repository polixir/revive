import torch
from typing import Dict


def next_ts_placeholder_policy_function(data : Dict[str, torch.Tensor]) -> torch.Tensor:
    action = data['placeholder']
    dim = action.shape[-1]
    ts_action = data['ts_'+'placeholder']
    return torch.cat([ts_action, action],axis=-1)[..., dim:]