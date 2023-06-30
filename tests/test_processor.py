import torch
import numpy as np
from revive.data.processor import DataProcessor

def test_processor():
    data_configs = {
        'obs' : [
            {
                'type' : 'category', 
                'dim' : 1,
                'values' : [0, 1, 2],
            },
            {
                'type' : 'discrete',
                'dim' : 2,
                'max' : [1, 2],
                'min' : [0, 0]
            },
            {
                'type' : 'continuous',
                'dim' : 2,
                'max' : [1, 2],
                'min' : [0, 0]
            },
        ]
    }

    process_parameters = {
        'obs' : {
            'forward_slices' : [slice(0, 1), slice(1, 3), slice(3, 5)],
            'backward_slices' : [slice(0, 3), slice(3, 5), slice(5, 7)],
            'additional_parameters' : [
                np.array([0, 1, 2]),
                (np.array([0.5, 1], dtype=np.float32), np.array([0.5, 1], dtype=np.float32), np.array([2, 5])),
                (np.array([0.5, 1], dtype=np.float32), np.array([0.5, 1], dtype=np.float32))
            ]
        }
    }

    orders = {
        'obs' : {
            'forward' : [0, 1, 2, 3, 4],
            'backward' : [0, 1, 2, 3, 4],
        }
    }

    processor = DataProcessor(data_configs, process_parameters, orders)

    original_numpy_data = {'obs' : np.array([0, 1, 1, 0.5, 1], dtype=np.float32)}
    processed_numpy_data = processor.process(original_numpy_data)
    deprocessed_numpy_data = processor.deprocess(processed_numpy_data)
    assert np.all(original_numpy_data['obs'] == deprocessed_numpy_data['obs']), f"{original_numpy_data['obs']}, {deprocessed_numpy_data['obs']}"

    original_torch_data = {'obs' : torch.tensor(original_numpy_data['obs'])}
    processed_torch_data = processor.process_torch(original_torch_data)
    deprocessed_torch_data = processor.deprocess_torch(processed_torch_data)
    assert torch.all(original_torch_data['obs'] == deprocessed_torch_data['obs']), f"{original_torch_data['obs']}, {deprocessed_torch_data['obs']}"