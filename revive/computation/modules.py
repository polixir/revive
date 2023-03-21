''''''
"""
    POLIXIR REVIVE, copyright (C) 2021-2023 Polixir Technologies Co., Ltd., is 
    distributed under the GNU Lesser General Public License (GNU LGPL). 
    POLIXIR REVIVE is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 3 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.
"""

import torch
import warnings
from torch import nn
from typing import Optional, Union, List, Dict
from collections import OrderedDict

from revive.computation.dists import *
from revive.computation.utils import *

ACTIVATION_CREATORS = {
    'relu' : lambda dim: nn.ReLU(inplace=True),
    'elu' : lambda dim: nn.ELU(),
    'leakyrelu' : lambda dim: nn.LeakyReLU(negative_slope=0.1, inplace=True),
    'tanh' : lambda dim: nn.Tanh(),
    'sigmoid' : lambda dim: nn.Sigmoid(),
    'identity' : lambda dim: nn.Identity(),
    'prelu' : lambda dim: nn.PReLU(dim),
    'gelu' : lambda dim: nn.GELU(),
    'swish' : lambda dim: Swish(),
}

class Swish(nn.Module):
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

# -------------------------------- Backbones -------------------------------- #
class MLP(nn.Module):
    r"""
        Multi-layer Perceptron

        Args:
        
            in_features : int, features numbers of the input

            out_features : int, features numbers of the output

            hidden_features : int, features numbers of the hidden layers

            hidden_layers : int, numbers of the hidden layers

            norm : str, normalization method between hidden layers, default : None

            hidden_activation : str, activation function used in hidden layers, default : 'leakyrelu'

            output_activation : str, activation function used in output layer, default : 'identity'
    """
    def __init__(self, 
                 in_features : int, 
                 out_features : int, 
                 hidden_features : int, 
                 hidden_layers : int, 
                 norm : str = None, 
                 hidden_activation : str = 'leakyrelu', 
                 output_activation : str = 'identity'):
        super(MLP, self).__init__()

        hidden_activation_creator = ACTIVATION_CREATORS[hidden_activation]
        output_activation_creator = ACTIVATION_CREATORS[output_activation]

        if hidden_layers == 0:
            self.net = nn.Sequential(
                nn.Linear(in_features, out_features),
                output_activation_creator(out_features)
            )
        else:
            net = []
            for i in range(hidden_layers):
                net.append(nn.Linear(in_features if i == 0 else hidden_features, hidden_features))
                if norm:
                    if norm == 'ln':
                        net.append(nn.LayerNorm(hidden_features))
                    elif norm == 'bn':
                        net.append(nn.BatchNorm1d(hidden_features))
                    else:
                        raise NotImplementedError(f'{norm} does not supported!')
                net.append(hidden_activation_creator(hidden_features))
            net.append(nn.Linear(hidden_features, out_features))
            net.append(output_activation_creator(out_features))
            self.net = nn.Sequential(*net)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        r""" forward method of MLP only assume the last dim of x matches `in_features` """
        return self.net(x)

class ResBlock(nn.Module):
    def __init__(self, input_feature : int, output_feature : int, norm : str = 'ln'):
        super().__init__()

        if norm == 'ln':
            norm_class = torch.nn.LayerNorm
            self.process_net = torch.nn.Sequential(
                torch.nn.Linear(input_feature, output_feature),
                norm_class(output_feature),
                torch.nn.ReLU(True),
                torch.nn.Linear(output_feature, output_feature),
                norm_class(output_feature),
                torch.nn.ReLU(True)
            )
        else:
            self.process_net = torch.nn.Sequential(
                torch.nn.Linear(input_feature, output_feature),
                torch.nn.ReLU(True),
                torch.nn.Linear(output_feature, output_feature),
                torch.nn.ReLU(True)
            )

        if not input_feature == output_feature:
            self.skip_net = torch.nn.Linear(input_feature, output_feature)
        else:
            self.skip_net = torch.nn.Identity()

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        '''x should be a 2D Tensor due to batchnorm'''
        return self.process_net(x) + self.skip_net(x)

class ResNet(torch.nn.Module):
    def __init__(self, 
                 in_features : int, 
                 out_features : int, 
                 hidden_features : int, 
                 hidden_layers : int, 
                 norm : str = 'bn',
                 output_activation : str = 'identity'):
        super().__init__()

        modules = []
        for i in range(hidden_layers):
            if i == 0:
                modules.append(ResBlock(in_features, hidden_features, norm))
            else:
                modules.append(ResBlock(hidden_features, hidden_features, norm))
        modules.append(torch.nn.Linear(hidden_features, out_features))
        modules.append(ACTIVATION_CREATORS[output_activation](out_features))

        self.resnet = torch.nn.Sequential(*modules)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        '''NOTE: reshape is needed since resblock only support 2D Tensor'''
        shape = x.shape
        x = x.view(-1, shape[-1])
        output = self.resnet(x)
        output = output.view(*shape[:-1], -1)
        return output 

class Transformer1D(nn.Module):
    def __init__(self, 
                 in_features : int,
                 out_features : int, 
                 transformer_features : int = 16, 
                 transformer_heads : int = 8, 
                 transformer_layers : int = 4):
        super().__init__()

        self.linear = torch.nn.Linear(in_features, out_features)

        self.register_parameter('in_weight', torch.nn.Parameter(torch.randn(out_features, transformer_features)))
        self.register_parameter('in_bais', torch.nn.Parameter(torch.zeros(out_features, transformer_features)))
        self.register_parameter('out_weight', torch.nn.Parameter(torch.randn(out_features, transformer_features)))
        self.register_parameter('out_bais', torch.nn.Parameter(torch.zeros(out_features)))

        torch.nn.init.xavier_normal_(self.in_weight)
        torch.nn.init.zeros_(self.out_weight) # zero initialize

        encoder_layer = torch.nn.TransformerEncoderLayer(transformer_features, transformer_heads, 512)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, transformer_layers)   

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.view(-1, x.shape[-1]) # [B, I]
        x = self.linear(x) # [B, O]
        x = x.unsqueeze(dim=-1) # [B, O, 1]
        x = x * self.in_weight + self.in_bais # [B, O, F]
        x = x.permute(1, 0, 2).contiguous() # [O, B, F]
        x = self.transformer(x) # [O, B, F]
        x = x.permute(1, 0, 2).contiguous() # [B, O, F]
        x = torch.sum(x * self.out_weight, dim=-1) + self.out_bais # [B, O]
        x = x.view(*shape[:-1], x.shape[-1])
        return x

class DistributionWrapper(nn.Module):
    r"""wrap output of Module to distribution"""
    BASE_TYPES = ['normal', 'gmm', 'onehot', 'discrete_logistic']
    SUPPORTED_TYPES = BASE_TYPES + ['mix']

    def __init__(self, distribution_type : str = 'normal', **params):
        super().__init__()
        self.distribution_type = distribution_type
        self.params = params

        assert self.distribution_type in self.SUPPORTED_TYPES, f"{self.distribution_type} is not supported!"

        if self.distribution_type == 'normal':
            self.max_logstd = nn.Parameter(torch.ones(self.params['dim']) * 0, requires_grad=True)
            self.min_logstd = nn.Parameter(torch.ones(self.params['dim']) * -10, requires_grad=True)
            if not self.params.get('conditioned_std', True):
                self.logstd = nn.Parameter(torch.zeros(self.params['dim']), requires_grad=True)
        elif self.distribution_type == 'gmm':
            self.max_logstd = nn.Parameter(torch.ones(self.params['mixture'], self.params['dim']) * 0, requires_grad=True)
            self.min_logstd = nn.Parameter(torch.ones(self.params['mixture'], self.params['dim']) * -10, requires_grad=True)            
            if not self.params.get('conditioned_std', True):
                self.logstd = nn.Parameter(torch.zeros(self.params['mixture'], self.params['dim']), requires_grad=True)
        elif self.distribution_type == 'discrete_logistic':
            self.num = self.params['num']
        elif self.distribution_type == 'mix':
            assert 'dist_config' in self.params.keys(), "You need to provide `dist_config` for Mix distribution"

            self.dist_config = self.params['dist_config']
            self.wrapper_list = []
            self.input_sizes = []
            self.output_sizes = []
            for config in self.dist_config:
                dist_type = config['type']
                assert dist_type in self.SUPPORTED_TYPES, f"{dist_type} is not supported!"
                assert not dist_type == 'mix', "recursive MixDistribution is not supported!"

                self.wrapper_list.append(DistributionWrapper(dist_type, **config))

                self.input_sizes.append(config['dim'])
                self.output_sizes.append(config['output_dim'])
                
            self.wrapper_list = nn.ModuleList(self.wrapper_list)                                     

    def forward(self, x : torch.Tensor, adapt_std : Optional[torch.Tensor] = None, payload : Optional[torch.Tensor] = None) -> ReviveDistribution:
        '''
            Warp the given tensor to distribution

            :param adapt_std : it will overwrite the std part of the distribution (optional)
            :param payload : payload will be applied to the output distribution after built (optional)
        '''
        # [ OTHER ] replace the clip with tanh
        # [ OTHER ] better std controlling strategy is needed todo
        if self.distribution_type == 'normal':
            if self.params.get('conditioned_std', True):
                mu, logstd = torch.chunk(x, 2, dim=-1)
                logstd_logit = logstd
                max_std = 0.5
                min_std = 0.001
                std = (torch.tanh(logstd_logit) + 1) / 2 * (max_std - min_std) + min_std
            else:
                mu, logstd = x, self.logstd
                logstd_logit = self.logstd
                max_std = 0.5
                min_std = 0.001
                std = (torch.tanh(logstd_logit) + 1) / 2 * (max_std - min_std) + min_std
                # std = torch.exp(logstd)
            if payload is not None:
                mu = mu + safe_atanh(payload)
            mu = torch.tanh(mu)
            # std = adapt_std if adapt_std is not None else torch.exp(soft_clamp(logstd, self.min_logstd.to(logstd.device), self.max_logstd.to(logstd.device)))
            return DiagnalNormal(mu, std)
        elif self.distribution_type == 'gmm':
            if self.params.get('conditioned_std', True):
                logits, mus, logstds = torch.split(x, [self.params['mixture'], 
                                                       self.params['mixture'] * self.params['dim'], 
                                                       self.params['mixture'] * self.params['dim']], dim=-1)
                mus = mus.view(*mus.shape[:-1], self.params['mixture'], self.params['dim'])      
                logstds = logstds.view(*logstds.shape[:-1], self.params['mixture'], self.params['dim'])
            else:
                logits, mus = torch.split(x, [self.params['mixture'], self.params['mixture'] * self.params['dim']], dim=-1)
                logstds = self.logstd
            if payload is not None:
                mus = mus + safe_atanh(payload.unsqueeze(dim=-2))
            mus = torch.tanh(mus)
            stds = adapt_std if adapt_std is not None else torch.exp(soft_clamp(logstds, self.min_logstd, self.max_logstd))
            return GaussianMixture(mus, stds, logits)
        elif self.distribution_type == 'onehot':
            return Onehot(x)
        elif self.distribution_type == 'discrete_logistic':
            mu, logstd = torch.chunk(x, 2, dim=-1)
            logstd = torch.clamp(logstd, -7, 1)
            return DiscreteLogistic(mu, torch.exp(logstd), num=self.num)
        elif self.distribution_type == 'mix':
            xs = torch.split(x, self.output_sizes, dim=-1)
            
            if adapt_std is not None:
                adapt_stds = torch.split(adapt_std, self.input_sizes, dim=-1)
            else:
                adapt_stds = [None] * len(self.input_sizes)
            
            if payload is not None:
                payloads = torch.split(payload, self.input_sizes + [payload.shape[-1] - sum(self.input_sizes)], dim=-1)[:-1]
            else:
                payloads = [None] * len(self.input_sizes)

            dists = [wrapper(x, _adapt_std, _payload) for x, _adapt_std, _payload, wrapper in zip(xs, adapt_stds, payloads, self.wrapper_list)]
            
            return MixDistribution(dists)

    def extra_repr(self) -> str:
        return 'type={}, dim={}'.format(
            self.distribution_type, 
            self.params['dim'] if not self.distribution_type == 'mix' else len(self.wrapper_list)
        )

# --------------------------------- Policies -------------------------------- #
class FeedForwardPolicy(torch.nn.Module):
    def __init__(self, 
                 in_features : Union[int, dict], 
                 out_features : int, 
                 hidden_features : int, 
                 hidden_layers : int, 
                 dist_config : list,
                 norm : str = None, 
                 hidden_activation : str = 'leakyrelu', 
                 backbone_type : Union[str, np.str_] = 'mlp',
                 use_multihead : bool = False,
                 use_feature_embed : bool = False):
        super().__init__()
        self.multihead = []
        self.dist_config = dist_config
        self.use_feature_embed = use_feature_embed
        self.feature_embed_layers = OrderedDict()

        if self.use_feature_embed:
            embed_dim = 64
            for input_name in sorted(in_features.keys()):
                self.feature_embed_layers[input_name] = MLP(in_features[input_name], embed_dim, 64, 1, norm=None, hidden_activation='leakyrelu')
            self.feature_embed_layers = nn.ModuleDict(self.feature_embed_layers)
            in_features = embed_dim * len(in_features.keys())

        if not use_multihead:
            if backbone_type == 'mlp':
                self.backbone = MLP(in_features, out_features, hidden_features, hidden_layers, norm, hidden_activation)
            elif backbone_type == 'res':
                self.backbone = ResNet(in_features, out_features, hidden_features, hidden_layers, norm)
            elif backbone_type == 'transformer':
                self.backbone = Transformer1D(in_features, out_features, hidden_features, transformer_layers=hidden_layers)
            else:
                raise NotImplementedError(f'backbone type {backbone_type} is not supported')
        else:
            if backbone_type == 'mlp':
                self.backbone = MLP(in_features, hidden_features, hidden_features, hidden_layers, norm, hidden_activation)
            elif backbone_type == 'res':
                self.backbone = ResNet(in_features, hidden_features, hidden_features, hidden_layers, norm)
            elif backbone_type == 'transformer':
                self.backbone = Transformer1D(in_features, hidden_features, hidden_features, transformer_layers=hidden_layers)
            else:
                raise NotImplementedError(f'backbone type {backbone_type} is not supported')
        
            
            for dist in self.dist_config:
                dist_type = dist["type"]
                output_dim = dist["output_dim"] 
                if dist_type == "onehot" or dist_type == "discrete_logistic":
                    self.multihead.append(MLP(hidden_features, output_dim, 64, 1, norm=None, hidden_activation='leakyrelu')) 
                elif dist_type == "normal":
                    normal_dim = dist["dim"]
                    for dim in range(normal_dim):  
                        self.multihead.append(MLP(hidden_features, int(output_dim // normal_dim), 64, 1, norm=None, hidden_activation='leakyrelu')) 
                else:
                    raise NotImplementedError(f'Dist type {dist_type} is not supported in multihead.')
                
            self.multihead = nn.ModuleList(self.multihead)

        self.dist_wrapper = DistributionWrapper('mix', dist_config=dist_config)
        

    def forward(self, state : Union[torch.Tensor, Dict[str, torch.Tensor]], adapt_std : Optional[torch.Tensor] = None, **kwargs) -> ReviveDistribution:
        if self.use_feature_embed:
            assert isinstance(state, dict), f"input must be dict, not {type(state)}, if you use node embedding"
            total_embeddings = []
            for key in self.feature_embed_layers:
                total_embeddings.append(self.feature_embed_layers[key](state[key]))
            state = torch.cat(total_embeddings, dim=-1)

        if not self.multihead:
            output = self.backbone(state)
        else:
            backbone_output = self.backbone(state) 
            multihead_output = []
            multihead_index = 0
            for dist in self.dist_config:
                dist_type = dist["type"]
                output_dim = dist["output_dim"] 
                if dist_type == "onehot" or dist_type == "discrete_logistic":
                    multihead_output.append(self.multihead[multihead_index](backbone_output))
                    multihead_index += 1
                elif dist_type == "normal":
                    normal_mode_output = []
                    normal_std_output = []
                    for head in self.multihead[multihead_index:]:
                        head_output = head(backbone_output)
                        if head_output.shape[-1] == 1:
                            mode = head_output
                            normal_mode_output.append(mode)
                        else:
                            mode, std = torch.chunk(head_output, 2, axis=-1)
                            normal_mode_output.append(mode)
                            normal_std_output.append(std)          
                    normal_output = torch.cat(normal_mode_output, axis=-1)
                    if normal_std_output:
                        normal_std_output = torch.cat(normal_std_output, axis=-1)
                        normal_output = torch.cat([normal_output, normal_std_output], axis=-1)
                        
                    multihead_output.append(normal_output)
                    break
                else:
                    raise NotImplementedError(f'Dist type {dist_type} is not supported in multihead.')                
            output = torch.cat(multihead_output, axis= -1)
                        
        dist = self.dist_wrapper(output, adapt_std)
        if hasattr(self, "dist_mu_shift"):
            dist = dist.shift(self.dist_mu_shift)

        return dist

    def reset(self):
        pass
    
    @torch.no_grad()
    def get_action(self, state : Union[torch.Tensor, Dict[str, torch.Tensor]], deterministic : bool = True):
        dist = self(state)
        return dist.mode if deterministic else dist.sample()


class RecurrentPolicy(torch.nn.Module):
    def __init__(self, 
                 in_features : int, 
                 out_features : int, 
                 hidden_features : int, 
                 hidden_layers : int, 
                 dist_config : list, 
                 backbone_type : Union[str, np.str_] ='gru'):
        super().__init__()

        RNN = torch.nn.GRU if backbone_type == 'gru' else torch.nn.LSTM

        self.rnn = RNN(in_features, hidden_features, hidden_layers)
        self.mlp = MLP(hidden_features, out_features, 0, 0)
        self.dist_wrapper = DistributionWrapper('mix', dist_config=dist_config)

    def reset(self):
        self.h = None

    def forward(self, x : torch.Tensor, adapt_std : Optional[torch.Tensor] = None, **kwargs) -> ReviveDistribution:
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            rnn_output, self.h = self.rnn(x, self.h)
            rnn_output = rnn_output.squeeze(0)
        else:
            rnn_output, self.h = self.rnn(x)
        logits = self.mlp(rnn_output)
        return self.dist_wrapper(logits, adapt_std)


class RecurrentMLPPolicy(torch.nn.Module):
    def __init__(self, 
                 in_features : int, 
                 out_features : int, 
                 hidden_features : int, 
                 hidden_layers : int, 
                 dist_config : list, 
                 backbone_type : Union[str, np.str_] ='gru',
                 rnn_hidden_features : int = 64,
                 joint_train : bool = False):
        super().__init__()

        self.joint_train = joint_train

        RNN = torch.nn.GRU if backbone_type == 'gru' else torch.nn.LSTM

        self.in_feature_embed = nn.Linear(in_features, hidden_features)
        self.rnn = RNN(in_features, rnn_hidden_features, hidden_layers)
        self.backbone = MLP(hidden_features + rnn_hidden_features, out_features, 0, 0)

        self.dist_wrapper = DistributionWrapper('mix', dist_config=dist_config)


    def reset(self):
        self.h = None

    def forward(self, x : torch.Tensor, adapt_std : Optional[torch.Tensor] = None, field: str = 'mail', **kwargs) -> ReviveDistribution:
        if field == 'bc':
            shape = x.shape
            if len(shape) == 2:
                x_embed = self.in_feature_embed(x)
                a = x.unsqueeze(0)
                rnn_output, self.h = self.rnn(a, self.h)
                rnn_output = rnn_output.squeeze(0)  # (bs, dim)
                logits = self.backbone(torch.concat([x_embed, rnn_output], dim=-1))  # (bs, dim)
            else:
                assert len(shape) == 3, f"expect len(x.shape) == 3. However got x.shape {shape}"
                self.h = None
                output = []
                for i in range(shape[0]):
                    a = x[i]
                    a = a.unsqueeze(0) #(1, bs, dim)
                    a_embed = self.in_feature_embed(a)  # (1, bs, dim)
                    rnn_output, self.h = self.rnn(a, self.h)  #(1, bs, dim)
                    backbone_output = self.backbone(torch.concat([a_embed, rnn_output], dim=-1))  # (1, bs, dim)
                    output.append(backbone_output)
                logits = torch.concat(output, dim=0)
        elif field == 'mail':
            shape = x.shape
            if len(shape) == 2:
                x_embed = self.in_feature_embed(x)
                a = x.unsqueeze(0)
                if self.joint_train:
                    rnn_output, self.h = self.rnn(a, self.h)
                else:
                    with torch.no_grad():
                        rnn_output, self.h = self.rnn(a, self.h)
                rnn_output = rnn_output.squeeze(0)  # (bs, dim)
                logits = self.backbone(torch.concat([x_embed, rnn_output], dim=-1))  # (bs, dim)
            else:
                assert len(shape) == 3, f"expect len(x.shape) == 3. However got x.shape {shape}"
                self.h = None
                output = []
                for i in range(shape[0]):
                    a = x[i]
                    a = a.unsqueeze(0) #(1, bs, dim)
                    a_embed = self.in_feature_embed(a)  # (1, bs, dim)
                    if self.joint_train:
                        rnn_output, self.h = self.rnn(a, self.h)  #(1, bs, dim)
                    else:
                        with torch.no_grad():
                            rnn_output, self.h = self.rnn(a, self.h)  #(1, bs, dim)
                    backbone_output = self.backbone(torch.concat([a_embed, rnn_output], dim=-1))  # (1, bs, dim)
                    output.append(backbone_output)
                logits = torch.concat(output, dim=0)
        return self.dist_wrapper(logits, adapt_std)


class RecurrentRESPolicy(torch.nn.Module):
    def __init__(self, 
                 in_features : int, 
                 out_features : int, 
                 hidden_features : int, 
                 hidden_layers : int, 
                 dist_config : list, 
                 backbone_type : Union[str, np.str_] ='gru',
                 rnn_hidden_features : int = 256,
                 joint_train : bool = False):
        super().__init__()

        self.joint_train = joint_train

        RNN = torch.nn.GRU if backbone_type == 'gru' else torch.nn.LSTM

        self.in_feature_embed = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            ACTIVATION_CREATORS['relu'](hidden_features),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_features)
        )

        self.side_net = nn.Sequential(
            ResBlock(hidden_features, hidden_features),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_features),
            ResBlock(hidden_features, hidden_features),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_features)
        )

        self.rnn = RNN(hidden_features, rnn_hidden_features, hidden_layers)
        self.backbone = MLP(hidden_features + rnn_hidden_features, out_features, hidden_features, 1)
        self.dist_wrapper = DistributionWrapper('mix', dist_config=dist_config)

    def reset(self):
        self.h = None

    def forward(self, x : torch.Tensor, adapt_std : Optional[torch.Tensor] = None, field: str = 'mail', **kwargs) -> ReviveDistribution:
        if field == 'bc':
            shape = x.shape
            if len(shape) == 2:
                x_embed = self.in_feature_embed(x)
                side_output = self.side_net(x_embed)
                a = x_embed.unsqueeze(0)
                rnn_output, self.h = self.rnn(a, self.h)
                rnn_output = rnn_output.squeeze(0)  # (bs, dim)
                logits = self.backbone(torch.concat([side_output, rnn_output], dim=-1))  # (bs, dim)
            else:
                assert len(shape) == 3, f"expect len(x.shape) == 3. However got x.shape {shape}"
                self.h = None
                output = []
                for i in range(shape[0]):
                    a = x[i]
                    a = a.unsqueeze(0) #(1, bs, dim)
                    a_embed = self.in_feature_embed(a)  # (1, bs, dim)
                    side_output = self.side_net(a_embed)
                    rnn_output, self.h = self.rnn(a_embed, self.h)  #(1, bs, dim)
                    backbone_output = self.backbone(torch.concat([side_output, rnn_output], dim=-1))  # (1, bs, dim)
                    output.append(backbone_output)
                logits = torch.concat(output, dim=0)
        elif field == 'mail':
            shape = x.shape
            if len(shape) == 2:
                x_embed = self.in_feature_embed(x)
                side_output = self.side_net(x_embed)
                a = x_embed.unsqueeze(0)
                if self.joint_train:
                    rnn_output, self.h = self.rnn(a, self.h)
                else:
                    with torch.no_grad():
                        rnn_output, self.h = self.rnn(a, self.h)
                rnn_output = rnn_output.squeeze(0)  # (bs, dim)
                logits = self.backbone(torch.concat([side_output, rnn_output], dim=-1))  # (bs, dim)
            else:
                assert len(shape) == 3, f"expect len(x.shape) == 3. However got x.shape {shape}"
                self.h = None
                output = []
                for i in range(shape[0]):
                    a = x[i]
                    a = a.unsqueeze(0) #(1, bs, dim)
                    a_embed = self.in_feature_embed(a)  # (1, bs, dim)
                    side_output = self.side_net(a_embed)
                    if self.joint_train:
                        rnn_output, self.h = self.rnn(a_embed, self.h)  #(1, bs, dim)
                    else:
                        with torch.no_grad():
                            rnn_output, self.h = self.rnn(a_embed, self.h)  #(1, bs, dim)
                    backbone_output = self.backbone(torch.concat([side_output, rnn_output], dim=-1))  # (1, bs, dim)
                    output.append(backbone_output)
                logits = torch.concat(output, dim=0)
        return self.dist_wrapper(logits, adapt_std)


class ContextualPolicy(torch.nn.Module):
    def __init__(self, 
                 in_features : int, 
                 out_features : int, 
                 hidden_features : int, 
                 hidden_layers : int, 
                 dist_config : list, 
                 backbone_type : Union[str, np.str_] ='contextual_gru'):
        super().__init__()

        RNN = torch.nn.GRU if backbone_type == 'contextual_gru' else torch.nn.LSTM
        self.preprocess_mlp = MLP(in_features, hidden_features, 0, 0, output_activation='leakyrelu')
        self.rnn = RNN(hidden_features, hidden_features, 1)
        self.mlp = MLP(hidden_features + in_features, out_features, hidden_features, hidden_layers)
        self.dist_wrapper = DistributionWrapper('mix', dist_config=dist_config)

    def reset(self):
        self.h = None

    def forward(self, x : torch.Tensor, adapt_std : Optional[torch.Tensor] = None, **kwargs) -> ReviveDistribution:
        in_feature = x
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            x = self.preprocess_mlp(x)
            rnn_output, self.h = self.rnn(x, self.h)
            rnn_output = rnn_output.squeeze(0)
        else:
            x = self.preprocess_mlp(x)
            rnn_output, self.h = self.rnn(x)
        logits = self.mlp(torch.cat((in_feature, rnn_output), dim=-1))
        return self.dist_wrapper(logits, adapt_std)

# ------------------------------- Transitions ------------------------------- #
class FeedForwardTransition(FeedForwardPolicy):
    def __init__(self, 
                 in_features : Union[int, dict], 
                 out_features : int, 
                 hidden_features : int, 
                 hidden_layers : int, 
                 dist_config : list,
                 norm : Optional[str] = None, 
                 hidden_activation : str = 'leakyrelu', 
                 backbone_type : Union[str, np.str_] = 'mlp',
                 mode : str = 'global',
                 obs_dim : Optional[int] = None,
                 use_feature_embed : bool = False):
        
        self.mode = mode
        self.obs_dim = obs_dim

        if self.mode == 'local': 
            dist_types = [config['type'] for config in dist_config]
            if 'onehot' in dist_types or 'discrete_logistic' in dist_types:
                warnings.warn('Detect distribution type that are not compatible with the local mode, fallback to global mode!')
                self.mode = 'global'

        if self.mode == 'local': assert self.obs_dim is not None, \
            "For local mode, the dim of observation should be given!"
        

        super(FeedForwardTransition, self).__init__(in_features, out_features, hidden_features, hidden_layers, 
                                                    dist_config, norm, hidden_activation, backbone_type, use_feature_embed=use_feature_embed)

    def reset(self):
        pass

    def forward(self, state : Union[torch.Tensor, Dict[str, torch.Tensor]], adapt_std : Optional[torch.Tensor] = None, **kwargs) -> ReviveDistribution:
        dist = super(FeedForwardTransition, self).forward(state, adapt_std)
        if self.mode == 'local' and self.obs_dim is not None:
            dist = dist.shift(state[..., :self.obs_dim])
        return dist


class RecurrentTransition(RecurrentPolicy):
    def __init__(self, 
                 in_features : int, 
                 out_features : int, 
                 hidden_features : int, 
                 hidden_layers : int, 
                 dist_config : list,
                 backbone_type : Union[str, np.str_] = 'mlp',
                 mode : str = 'global',
                 obs_dim : Optional[int] = None):
        
        self.mode = mode
        self.obs_dim = obs_dim

        if self.mode == 'local': assert not 'onehot' in [config['type'] for config in dist_config], \
            "The local mode of transition is not compatible with onehot data! Please fallback to global mode!"
        if self.mode == 'local': assert self.obs_dim is not None, \
            "For local mode, the dim of observation should be given!"

        super(RecurrentTransition, self).__init__(in_features, out_features, hidden_features, hidden_layers, 
                                                  dist_config, backbone_type)

    def forward(self, state : torch.Tensor, adapt_std : Optional[torch.Tensor] = None) -> ReviveDistribution:
        dist = super(RecurrentTransition, self).forward(state, adapt_std)
        if self.mode == 'local' and self.obs_dim is not None:
            dist = dist.shift(state[..., :self.obs_dim])
        return dist


class RecurrentMLPTransition(RecurrentMLPPolicy):
    def __init__(self, 
                 in_features : int, 
                 out_features : int, 
                 hidden_features : int, 
                 hidden_layers : int, 
                 dist_config : list,
                 backbone_type : Union[str, np.str_] = 'mlp',
                 mode : str = 'global',
                 obs_dim : Optional[int] = None,
                 rnn_hidden_features : int = 64,
                 joint_train : bool = False):
        
        self.mode = mode
        self.obs_dim = obs_dim

        if self.mode == 'local': assert not 'onehot' in [config['type'] for config in dist_config], \
            "The local mode of transition is not compatible with onehot data! Please fallback to global mode!"
        if self.mode == 'local': assert self.obs_dim is not None, \
            "For local mode, the dim of observation should be given!"

        super(RecurrentMLPTransition, self).__init__(in_features, out_features, hidden_features, hidden_layers, 
                                                  dist_config, backbone_type, joint_train=joint_train, rnn_hidden_features=rnn_hidden_features)

    def forward(self, state : torch.Tensor, adapt_std : Optional[torch.Tensor] = None, field: str = 'mail', **kwargs) -> ReviveDistribution:
        dist = super(RecurrentMLPTransition, self).forward(state, adapt_std, field)
        if self.mode == 'local' and self.obs_dim is not None:
            dist = dist.shift(state[..., :self.obs_dim])
        return dist


class RecurrentRESTransition(RecurrentRESPolicy):
    def __init__(self, 
                 in_features : int, 
                 out_features : int, 
                 hidden_features : int, 
                 hidden_layers : int, 
                 dist_config : list,
                 backbone_type : Union[str, np.str_] = 'mlp',
                 mode : str = 'global',
                 obs_dim : Optional[int] = None,
                 rnn_hidden_features : int = 64,
                 joint_train : bool = False):
        
        self.mode = mode
        self.obs_dim = obs_dim

        if self.mode == 'local': assert not 'onehot' in [config['type'] for config in dist_config], \
            "The local mode of transition is not compatible with onehot data! Please fallback to global mode!"
        if self.mode == 'local': assert self.obs_dim is not None, \
            "For local mode, the dim of observation should be given!"

        super(RecurrentRESTransition, self).__init__(in_features, out_features, hidden_features, hidden_layers, 
                                                  dist_config, backbone_type, joint_train=joint_train, rnn_hidden_features=rnn_hidden_features)

    def forward(self, state : torch.Tensor, adapt_std : Optional[torch.Tensor] = None, field: str = 'mail', **kwargs) -> ReviveDistribution:
        dist = super(RecurrentRESTransition, self).forward(state, adapt_std, field)
        if self.mode == 'local' and self.obs_dim is not None:
            dist = dist.shift(state[..., :self.obs_dim])
        return dist


class FeedForwardMatcher(torch.nn.Module):
    def __init__(self, 
                 in_features : int, 
                 hidden_features : int, 
                 hidden_layers : int, 
                 hidden_activation : str = 'leakyrelu', 
                 norm : str = None,
                 backbone_type : Union[str, np.str_] = 'mlp'):
        super().__init__()

        if backbone_type == 'mlp':
            self.backbone = MLP(in_features, 1, hidden_features, hidden_layers, norm, hidden_activation, output_activation='sigmoid')
        elif backbone_type == 'res':
            self.backbone = ResNet(in_features, 1, hidden_features, hidden_layers, norm, output_activation='sigmoid')
        elif backbone_type == 'transformer':
            self.backbone = torch.nn.Sequential(
                Transformer1D(in_features, in_features, hidden_features, transformer_layers=hidden_layers),
                torch.nn.Linear(in_features, 1),
                torch.nn.Sigmoid(),
            ) 
        else:
            raise NotImplementedError(f'backbone type {backbone_type} is not supported')

    def forward(self, *inputs : List[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(inputs, dim=-1)
        return self.backbone(x)

class RecurrentMatcher(torch.nn.Module):
    def __init__(self, 
                 in_features : int, 
                 hidden_features : int, 
                 hidden_layers : int, 
                 backbone_type : Union[str, np.str_] = 'gru', 
                 bidirect : bool = False):
        super().__init__()

        RNN = torch.nn.GRU if backbone_type == 'gru' else torch.nn.LSTM

        self.rnn = RNN(in_features, hidden_features, hidden_layers, bidirectional=bidirect)

        self.output_layer = MLP(hidden_features * (2 if bidirect else 1), 1, 0, 0, output_activation='sigmoid')

    def forward(self, *inputs : List[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(inputs, dim=-1)
        rnn_output = self.rnn(x)[0]
        return self.output_layer(rnn_output)

class HierarchicalMatcher(torch.nn.Module):
    def __init__(self, 
                 in_features : list, 
                 hidden_features : int, 
                 hidden_layers : int, 
                 hidden_activation : int, 
                 norm : str = None):
        super().__init__()
        self.in_features = in_features
        process_layers = []
        output_layers = []
        feature = self.in_features[0]
        for i in range(1, len(self.in_features)):
            feature += self.in_features[i]
            process_layers.append(MLP(feature, hidden_features, hidden_features, hidden_layers, norm, hidden_activation, hidden_activation))
            output_layers.append(torch.nn.Linear(hidden_features, 1))
            feature = hidden_features
        self.process_layers = torch.nn.ModuleList(process_layers)
        self.output_layers = torch.nn.ModuleList(output_layers)

    def forward(self, *inputs : List[torch.Tensor]) -> torch.Tensor:
        assert len(inputs) == len(self.in_features)
        last_feature = inputs[0]
        result = 0
        for i, process_layer, output_layer in zip(range(1, len(self.in_features)), self.process_layers, self.output_layers):
            last_feature = torch.cat([last_feature, inputs[i]], dim=-1)
            last_feature = process_layer(last_feature)
            result += output_layer(last_feature)
        return torch.sigmoid(result)