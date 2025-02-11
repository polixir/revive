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
import numpy as np
from typing import Dict

from revive.data.batch import Batch


class DataProcessor:
    """
        This class deal with the data mapping between original format and the computation format.

        There are two steps for mapping from original to computation:

        Step 1: Reorder the data. This is to group variables with the same type to accelerate computation.

        Step 2: If the variable is continuous or discrete, normalize the data to [-1, 1].
                If the variable is categorical, create an onehot vector.

        Mapping from computation to original is the reverse of these steps.
        
        Args:
            data_configs (dict): A dictionary containing the configuration of the input data.
            processing_params (dict): A dictionary containing the processing parameters.
            orders (list): A list of variable orders for reordering the data.
    """

    def __init__(self, data_configs, processing_params, orders):
        self.data_configs = data_configs
        self.processing_params = processing_params
        self.orders = orders

    @property
    def keys(self):
        return list(self.data_configs.keys())

    # ----------------------------------------------------------------------------------- #
    #                                Fuctions for Tensor                                  # 

    def _process_fn_torch(self, data : torch.Tensor, data_config, processing_params, order):
        """This method applies the transformation to the input data tensor."""
        data = data[..., order['forward']]
        processed_data = []
        for config, s, param in zip(data_config, processing_params['forward_slices'], processing_params['additional_parameters']):
            _data = data[..., s]
            if config['type'] == 'category':
                values = torch.tensor(param.copy()).to(_data)
                onehot = (_data == values).float()
                processed_data.append(onehot)
            elif config['type'] == 'continuous':
                mean, std = param
                mean = torch.tensor(mean.copy(), dtype=_data.dtype, device=_data.device)
                std = torch.tensor(std.copy(), dtype=_data.dtype, device=_data.device)
                _data = (_data - mean) / std
                processed_data.append(_data)
            elif config['type'] == 'discrete':
                mean, std, num = param
                mean = torch.tensor(mean.copy(), dtype=_data.dtype, device=_data.device)
                std = torch.tensor(std.copy(), dtype=_data.dtype, device=_data.device)  
                _data = (_data - mean) / std   
                processed_data.append(_data) 

        return torch.cat(processed_data, dim=-1)

    def _deprocess_fn_torch(self, data : torch.Tensor, data_config, processing_params, order):
        """This method applies the inverse transformation to the input data tensor to obtain the original data."""
        processed_data = []
        for config, s, param in zip(data_config, processing_params['backward_slices'], processing_params['additional_parameters']):
            _data = data[..., s]
            if config['type'] == 'category':
                values = torch.tensor(param.copy()).to(_data)
                _data = values[torch.argmax(_data, axis=-1)].float().to(data)
                _data = _data.unsqueeze(-1)
                processed_data.append(_data)
            elif config['type'] == 'continuous':
                mean, std = param
                mean = torch.tensor(mean.copy(), dtype=_data.dtype, device=_data.device)
                std = torch.tensor(std.copy(), dtype=_data.dtype, device=_data.device)
                _data = _data * std + mean
                processed_data.append(_data)
            elif config['type'] == 'discrete':
                mean, std, num = param
                mean = torch.tensor(mean.copy(), dtype=_data.dtype, device=_data.device)
                std = torch.tensor(std.copy(), dtype=_data.dtype, device=_data.device)
                num = torch.tensor(num.copy(), dtype=_data.dtype, device=_data.device)
                _data = (_data + 1) / 2 * (num - 1)
                _data = torch.round(_data) / (num - 1) * 2 - 1 
                _data = _data * std + mean      
                processed_data.append(_data) 

        processed_data = torch.cat(processed_data, axis=-1)   
        processed_data = processed_data[..., order['backward']]
        return processed_data  

    def process_single_torch(self, data : torch.Tensor, key: str) -> torch.Tensor:
        """
        Preprocess single data according different types of data including 'category', 'continuous', and 'discrete'.

        """
        if key in self.keys:
            return self._process_fn_torch(data, self.data_configs[key], self.processing_params[key], self.orders[key])
        else:
            return data

    def deprocess_single_torch(self, data : torch.Tensor, key: str) -> torch.Tensor:
        """
        Post process single data according different types of data including 'category', 'continuous', and 'discrete'.

        """
        if key in self.keys:
            return self._deprocess_fn_torch(data, self.data_configs[key], self.processing_params[key], self.orders[key])
        else:
            return data

    def process_torch(self, data):
        """
        Preprocess batch data according different types of data including 'category', 'continuous', and 'discrete'.

        """
        return Batch({k : self.process_single_torch(data[k], k) for k in data.keys()})
    
    def deprocess_torch(self, data):
        """
            Post process batch data according different types of data including 'category', 'continuous', and 'discrete'.
        """
        return Batch({k : self.deprocess_single_torch(data[k], k) for k in data.keys()})

    # ----------------------------------------------------------------------------------- #
    #                                Fuctions for ndarray                                 # 

    def _process_fn(self, data : np.ndarray, data_config, processing_params, order):
        """This method applies the transformation to the input data array."""
        data = data.take(order['forward'], axis=-1)
        processed_data = []
        for config, s, param in zip(data_config, processing_params['forward_slices'], processing_params['additional_parameters']):
            _data = data[..., s]
            if config['type'] == 'category':
                values = param
                onehot = (_data == values).astype(np.float32)
                assert np.all(onehot.sum(axis=-1) == 1), f'{onehot}, {values}, {_data}'
                processed_data.append(onehot)
            elif config['type'] == 'continuous':
                mean, std = param
                _data = (_data - mean) / std
                processed_data.append(_data)
            elif config['type'] == 'discrete':
                mean, std, num = param 
                _data = (_data - mean) / std   
                processed_data.append(_data) 

        return np.concatenate(processed_data, axis=-1)

    def _deprocess_fn(self, data : np.ndarray, data_config, processing_params, order):
        """This method applies the inverse transformation to the input data array to obtain the original data."""
        processed_data = []
        for config, s, param in zip(data_config, processing_params['backward_slices'], processing_params['additional_parameters']):
            _data = data[..., s]
            if config['type'] == 'category':
                values = param
                _data = values[np.argmax(_data, axis=-1)].astype(np.float32)
                _data = _data.reshape([*_data.shape, 1])
                processed_data.append(_data)
            elif config['type'] == 'continuous':
                mean, std = param
                _data = _data * std + mean
                processed_data.append(_data)
            elif config['type'] == 'discrete':
                mean, std, num = param
                _data = (_data + 1) / 2 * (num - 1)
                _data = np.round(_data) / (num - 1) * 2 - 1 
                _data = _data * std + mean      
                processed_data.append(_data) 

        processed_data = np.concatenate(processed_data, axis=-1)   
        processed_data = processed_data.take(order['backward'], axis=-1)
        return processed_data  

    def process_single(self, data : np.ndarray, key: str) -> np.ndarray:
        """
            Preprocess single data according different types of data including 'category', 'continuous', and 'discrete'.
        """
        if key in self.keys:
            return self._process_fn(data, self.data_configs[key], self.processing_params[key], self.orders[key])
        else:
            return data

    def deprocess_single(self, data : np.ndarray, key: str) -> np.ndarray:
        """
            Post process single data according different types of data including 'category', 'continuous', and 'discrete'.
        """
        if key in self.keys:
            return self._deprocess_fn(data, self.data_configs[key], self.processing_params[key], self.orders[key])
        else:
            return data

    def process(self, data : Dict[str, np.ndarray]):
        """
            Preprocess batch data according different types of data including 'category', 'continuous', and 'discrete'.
        """
        return Batch({k : self.process_single(data[k], k) for k in data.keys()})
    
    def deprocess(self, data : Dict[str, np.ndarray]):
        """
            Post process batch data according different types of data including 'category', 'continuous', and 'discrete'.
        """
        return Batch({k : self.deprocess_single(data[k], k) for k in data.keys()})
