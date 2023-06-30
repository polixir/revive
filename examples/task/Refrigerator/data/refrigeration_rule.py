import torch
from copy import deepcopy
from typing import Dict


normalize = False
weight = 1.0

def get_reward(data : Dict[str, torch.Tensor], graph) -> torch.Tensor:
    '''
        data: contains all the node defined in the computational graph.
        return reward
    '''
    noise_data = deepcopy(data)    
    noise_data["action"] += torch.randn_like(noise_data["action"]) * 0.1
    
    node_name = "next_temperature"
    
    if graph.get_node(node_name).node_type == 'network':
        node_output = graph.compute_node(node_name, noise_data).mode
    else:
        node_output = graph.compute_node(node_name, current_batch)
        
    correlation = ((noise_data["action"] - data["action"]) * (node_output-data[node_name]))
    
    reward = torch.where(correlation> 0, -0.2 * torch.ones_like(correlation[...,:1]), torch.zeros_like(correlation[...,:1]))

    return reward