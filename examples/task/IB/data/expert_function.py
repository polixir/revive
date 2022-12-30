import torch
from typing import Dict


def next_obs(data: Dict[str, torch.Tensor]) -> torch.Tensor:
   obs = data["obs"]
   current_next_obs = data["current_next_obs"]

   next_obs = torch.cat([current_next_obs, obs],axis=-1)[...,:obs.shape[-1]]
   
   return next_obs