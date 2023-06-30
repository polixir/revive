import torch
from typing import Dict

def mae_loss(kwargs) -> torch.Tensor:  
    node_name = kwargs["node_name"]
    node_dist = kwargs["node_dist"]
    expert_data = kwargs["expert_data"]
    graph = kwargs["graph"]
    
    # get network output data -> node_dist.mode
    # get node expert data -> expert_data[node_name]
    # reverse normalization data -> graph.nodes[node_name].processor.deprocess_torch({node_name:expert_data[node_name]})
    policy_loss = (node_dist.mode - expert_data[node_name]).abs().sum(dim=-1).mean()
        
    return policy_loss