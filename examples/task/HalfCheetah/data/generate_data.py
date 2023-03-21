import h5py
import numpy as np


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


if __name__ == "__main__":
    # 加载 halfcheetah-medium-v2 数据集
    # 请把 data_path 替换成对应的数据集路径
    data_path = "/home/ubuntu/.d4rl/datasets/halfcheetah_medium-v2.hdf5"
    data_dict = {}
    with h5py.File(data_path, 'r') as dataset_file:
        for k in get_keys(dataset_file):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]
    data = {key: data_dict[key] for key in ['actions','next_observations', 'observations', 'rewards', 'terminals', 'timeouts']}

    # recover delta_x from reward 
    timestep = 0.002
    frame_skip = 4
    dt = 0.05
    healthy_reward = 1
    ctrl_cost_weight = 0.1

    ctrl_cost = ctrl_cost_weight * np.sum(np.square(data['actions']),axis=-1, keepdims=True)
    forward_reward = data['rewards'].reshape(-1,1) + ctrl_cost
    data['delta_x'] = forward_reward * dt 
    data['done'] = data['terminals']*1

    # generate index info
    index = (np.where(np.sum((data['next_observations'][:-1] - data['observations'][1:]),axis=-1)!=0)[0]+1).tolist()+[data['observations'].shape[0]]
    index = [0]+index
    start = index[:-1]
    end = index[1:]
    traj = np.array(end) - np.array(start)

    # generate dataset for REVIVE
    outdata = {}
    outdata['obs'] = data['observations']
    outdata['next_obs'] = data['next_observations']
    outdata['action'] = data['actions']
    outdata['delta_x'] = data['delta_x']

    outdata['reward'] = data['rewards'].reshape(-1,1)
    outdata['done'] = data['done'].reshape(-1,1)
    outdata['done'] = (outdata['done']*1.0)

    outdata['index'] = (np.where(np.sum((outdata['next_obs'][:-1] - outdata['obs'][1:]),axis=-1)!=0)[0]+1).tolist() \
                        + [outdata['obs'].shape[0]]

    outdata_file = np.savez_compressed("halfcheetah-medium-v2.npz", **outdata)
