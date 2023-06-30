import torch

from revive.computation.modules import MLP, DistributionWrapper
from revive.computation.graph import NetworkDecisionNode

class EngineNetwork(torch.nn.Module):
    def __init__(self, 
                 input_dim,
                 output_dim,
                 hidden_features,
                 hidden_layers,
                 dist_config):
        super().__init__()

        self.engine_network_0 = MLP(input_dim, output_dim, hidden_features, hidden_layers)
        self.engine_network_1 = MLP(input_dim, output_dim, hidden_features, hidden_layers)
        self.engine_network_2 = MLP(input_dim, output_dim, hidden_features, hidden_layers)

        self.wrapper = self.dist_wrapper = DistributionWrapper('mix', dist_config=dist_config)

    def forward(self, obs, engine_index):
        engine_action_0 = self.engine_network_0(obs)
        engine_action_1 = self.engine_network_1(obs)
        engine_action_2 = self.engine_network_2(obs)

        engine_actions = torch.stack([engine_action_0, engine_action_1, engine_action_2], dim=-1)
        engine_index = engine_index.unsqueeze(dim=-2)
        engine_action = torch.sum(engine_actions * engine_index, dim=-1)

        return self.wrapper(engine_action)

class EngineNode(NetworkDecisionNode):
    def initialize_network(self, 
                           input_dim: int, 
                           output_dim: int, 
                           hidden_features: int, 
                           hidden_layers: int, 
                           hidden_activation: str, 
                           norm: str, 
                           backbone_type: str, 
                           dist_config: list, 
                           is_transition: bool, 
                           *args, **kwargs):
        
        self.network = EngineNetwork(5, output_dim, hidden_features, hidden_layers, dist_config)

    def __call__(self, data, *args, **kwargs):
        obs = data['obs']
        engine_index = data['engine_index'] # this is already onehot vector since it is defined as category
        output_dist = self.network(obs, engine_index) 
        return output_dist

    def reset(self) -> None:
        pass # pass reset since the network is stateless