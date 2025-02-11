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
import ot
import os
import ray
import gym
import yaml
import h5py
import torch
import math
import random
import urllib
import pickle
import argparse
import warnings
import importlib
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

from uuid import uuid1
from sklearn import tree
from cairosvg import svg2pdf
from dtreeviz.trees import dtreeviz
from tempfile import TemporaryDirectory
from PyPDF2 import PdfFileReader,PdfFileMerger
from tqdm import tqdm
from loguru import logger
from copy import deepcopy
from functools import partial
from torch.utils.data.dataloader import DataLoader
from typing import Any, Dict, List
from collections import defaultdict, deque
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy import stats

from revive.computation.graph import DesicionGraph
from revive.computation.inference import *
from revive.computation.utils import *
from revive.computation.modules import *
from revive.data.batch import Batch
from revive.computation.funs_parser import parser
# from revive.utils.common_utils import load_data

try:
    import cupy as cp
    CUPY_READY = True
except:
    # warnings.warn("Warning: CuPy is not installed, metric computing is going to be slow!")
    CUPY_READY = False


def update_env_vars(key: str, value: Any):
    """
    update env vars in os

    Args:
        key (str): name of the key.
        value (str): value for the key
    Returns:
        update os.environ['env_vars']
    """
    env_vars = os.environ.get('env_vars')
    if env_vars:
        try:
            env_vars_dict = eval(env_vars)
            if not isinstance(env_vars_dict, dict):
                raise ValueError('env_vars is not a dictionary')
        except (NameError, SyntaxError, ValueError) as e:
            return False
    else:
        env_vars_dict = {}

    env_vars_dict[key] = value
    os.environ['env_vars'] = str(env_vars_dict)


def get_env_var(key: str, default=None):
    """
    get env vars in os

    Args:
        key (str): name of the key.
        default (str): None
    Returns:
        update os.environ['env_vars']
    """

    env_vars = os.environ.get('env_vars')
    if env_vars:
        try:
            env_vars_dict = eval(env_vars)
            if not isinstance(env_vars_dict, dict):
                raise ValueError('vnv_vars is not a dictionary')
        except (NameError, SyntaxError, ValueError) as e:
            return default
    else:
        return default
    # return env_vars_dict.get(key, default)


class AttributeDict(dict):
    """
    define a new class for using get and set variables esily
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def setup_seed(seed: int):
    """
    Seting random seed in REVIVE.

    Args:
        seed: random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def load_npz(filename: str):
    """
    Loading npz file

    Args:
        filename(str): *.npz file path
    Return:
        Dict of data in format of keys:values
    """
    data = np.load(filename)
    return {k: v for k, v in data.items()}


def load_h5(filename: str):
    """
    Loading npz file

    Args:
        filename(str): *.h5 file path
    Return:
        Dict of data in format of keys:values
    """
    f = h5py.File(filename, 'r', libver="latest", swmr=True)
    data = {k: f[k][:] for k in f.keys()}
    f.close()
    return data


def save_h5(filename: str,
            data: Dict[str, np.ndarray]):
    """
    Loading npz file

    Args:
        filename(str): output *.h5 file path
    Return:
        output file
    """

    with h5py.File(filename, 'w') as f:
        for k, v in data.items():
            f[k] = v


def npz2h5(npz_filename: str,
           h5_filename: str):
    """
    Transforming npz file to h5 file

    Args:
        npz_filename (str): *.npz file path
        h5_filename (str): output *.h5 file path
    Return:
        output file
    """
    data = load_npz(npz_filename)
    save_h5(h5_filename, data)


def h52npz(h5_filename: str,
           npz_filename: str):
    """
    Transforming h5 file to npz file

    Args:
        h5_filename (str): input *.h5 file path
        npz_filename (str): output *.npz file path
    Return:
        output file
    """

    data = load_h5(h5_filename)
    np.savez_compressed(npz_filename, **data)


def load_data(data_file: str):
    """
    Loading data file
    Only support h5 and npz file as data files in REVIVE
    Args:
        data_file (str): input *.h5 or *.npz file path
    Return:
        Dict of data in format of keys:values
    """

    if data_file.endswith('.h5'):
        raw_data = load_h5(data_file)
    elif data_file.endswith('.npz'):
        raw_data = load_npz(data_file)
    else:
        raise ValueError(f'Try to load {data_file}, but get unknown data format!')
    return raw_data


def find_policy_index(graph: DesicionGraph,
                      policy_name: str):
    """
    Find index of policy node in the whole decision flow graph
    Args:
        graph (DesicionGraph): decision flow graph in REVIVE
        policy_name (str): the policy node name be indexed
    Return:
        index of the policy node in decision graph
    Notice:
        only the first policy node name is supported
        TODO: multi policy indexes
    """

    for i, k in enumerate(graph.keys()):
        if k == policy_name:
            break
    return i


def load_policy(filename: str,
                policy_name: str = None):
    """
    Load policy file for REVIVE in the format of torch or .pkl
    of VirturalEnv VirtualEnvDev or PolicyModelEv
    Args:
        filename (str): file path
        policy_name (str): the policy node name be indexed
    Return:
        Policy model
    """

    try:
        model = torch.load(filename, map_location='cpu')
    except:
        with open(filename, 'rb') as f:
            model = pickle.load(f)

    if isinstance(model, VirtualEnv):
        model = model._env

    if isinstance(model, VirtualEnvDev):
        node = model.graph.get_node(policy_name)
        model = PolicyModelDev(node)

    if isinstance(model, PolicyModelDev):
        model = PolicyModel(model)

    return model


def download_helper(url: str, filename: str):
    """
    Download file from given url. Modified from `torchvision.dataset.utils
    Args:
        url (str): donwloading path
        filename (str): output file path
    Return:
        Output path
    """
    def gen_bar_updater():
        pbar = tqdm(total=None)

        def bar_update(count, block_size, total_size):
            if pbar.total is None and total_size:
                pbar.total = total_size
            progress_bytes = count * block_size
            pbar.update(progress_bytes - pbar.n)

        return bar_update

    try:
        print('Downloading ' + url + ' to ' + filename)
        urllib.request.urlretrieve(
            url, filename,
            reporthook=gen_bar_updater()
        )
    except (urllib.error.URLError, IOError) as e:
        if url[:5] == 'https':
            url = url.replace('https:', 'http:')
            print('Failed download. Trying https -> http instead.',
                  ' Downloading ' + url + ' to ' + filename)
            urllib.request.urlretrieve(
                url, filename,
                reporthook=gen_bar_updater()
            )
        else:
            raise e


def import_module_from_file(file_path: str, module_name="module.name"):
    """
    import expert function from file
    Args:
        file_path (str): file path of the expert function
        module_name (str): function name in the file
    Return:
        treat the expert function as an useable funtion in REVIVE
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return foo


def get_reward_fn(reward_file_path: str, config_file: str):
    """
    import user defined reward function only for Matcher reward
    Args:
        reward_file_path (str): file path of the expert function
        config_file (str): decision flow *.yml file
    Return:
        treat the reward function as an useable funtion in REVIVE
    """
    if reward_file_path:
        logger.info(f'import reward function from {reward_file_path}!')
        # parse function
        reward_file_path_parsed = reward_file_path[:-3]+"_parsed.py"
        if not parser(reward_file_path, reward_file_path_parsed, config_file):
            reward_file_path_parsed = reward_file_path
        source_file = import_module_from_file(reward_file_path_parsed)
        try:
            reward_func = source_file.reward
        except:
            reward_func = source_file.get_reward
    else:
        logger.info(f'No reward function is defined!')
        reward_func = None

    return reward_func


def get_module(function_file_path, config_file):
    """
    import user defined function
    Args:
        function_file_path (str): file path of the expert function
        config_file (str): decision flow *.yml file
    Return:
        treat the reward function as an useable funtion in REVIVE
    """
    if function_file_path:
        logger.info(f'import module from {function_file_path}!')
        # parse function
        function_file_path_parsed = function_file_path[:-3]+"_parsed.py"
        if not parser(function_file_path,function_file_path_parsed,config_file):
            function_file_path_parsed = function_file_path
        module = import_module_from_file(function_file_path_parsed)

    else:
        logger.info(f'No module is defined!')
        module = None

    return module


def create_env(task: str):
    """
    initiating gym environment as testing env for trainning
    Args:
        task (str): gym mujoco task name
    Return:
        gym env
    """
    try:
        if task in ["HalfCheetah-v3", "Hopper-v3", "Walker2d-v3",
                    "ib", "finance", "citylearn"]:
            import neorl
            env = neorl.make(task)
        elif task in ['halfcheetah-meidum-v0', 'hopper-medium-v0',
                      'walker2d-medium-v0']:
            import d4rl
            env = gym.make(task)
        else:
            env = gym.make(task)
    except:
        warnings.warn(f'Warning: task {task} can not be created!')
        env = None

    return env


def test_one_trail(env: gym.Env, policy: PolicyModel):
    """
    testing revive policy on gym env
    Args:
        env (str): initialized gym mujoco env
        policy (str): revive policy used for testing on the env
    Return:
        reward and running length of the policy
    """
    env = deepcopy(env)
    policy = deepcopy(policy)

    obs = env.reset()
    reward = 0
    length = 0
    while True:
        action = policy.infer({'obs': obs[np.newaxis]})[0]
        obs, r, done, info = env.step(action)
        reward += r
        length += 1

        if done:
            break

    return (reward, length)


def test_on_real_env(env: gym.Env, policy: PolicyModel,
                     number_of_runs: int = 10):
    """
    testing revive policy on multiple gym envs
    Args:
        env (str): initialized gym mujoco env
        policy (str): revive policy used for testing on the env
        number_of_runs (int): the number of trails to testing
    Return:
        mean value of reward and running length of the policy
    """
    rewards = []
    episode_lengths = []
    test_func = ray.remote(test_one_trail)

    results = ray.get([test_func.remote(env, policy) for _ in range(number_of_runs)])
    rewards = [result[0] for result in results]
    episode_lengths = [result[1] for result in results]

    return np.mean(rewards), np.mean(episode_lengths)


def get_input_dim_from_graph(graph: DesicionGraph,
                             node_name: str,
                             total_dims: dict):
    """
    return the total number of dims used to compute the given node on the graph
    Args:
        graph (DecisionGraph): decision flow with user setting nodes
        node_name (str): name of the node to get total dimensions
        total_dims (dict): dict of input and output dims of all nodes
    Return:
        total number of dimensions of the node_name
    """

    input_names = graph[node_name]
    input_dim = 0
    for _name in input_names:
        input_dim += total_dims[_name]['input']
    return input_dim


def get_input_dim_dict_from_graph(graph: DesicionGraph, 
                                  node_name: str, 
                                  total_dims: dict):

    """
    return the total number of dims as dictused to compute the given node on the graph
    Args:
        graph (DecisionGraph): decision flow with user setting nodes
        node_name (str): name of the node to get total dimensions
        total_dims (dict): dict of input and output dims of all nodes
    Return:
        total number of dimensions as dict for all input of the node_name
    """

    input_names = graph[node_name]
    input_dim_dict = dict()
    for _name in input_names:
        input_dim_dict[_name] = total_dims[_name]['input']
    return input_dim_dict


def normalize(data: np.ndarray):
    """
    normalization of data using mean and std
    Args:
        data (np.ndarray): numpy array
    Return:
        normalized data
    """
    flatten_data = data.reshape((-1, data.shape[-1]))
    mean = flatten_data.mean(axis=0)
    std = flatten_data.std(axis=0)
    std[np.isclose(std, 0)] = 1
    data = (data - mean) / std
    return data


def plot_traj(traj: dict):
    """
    plot all dims of data into color map along trajectory
    Args:
        traj (dict): data stored in dict
    Return:
        plot show with x axis as dims and y axis as traj-step
    """
    traj = np.concatenate([*traj.values()], axis=-1)
    max_value = traj.max(axis=0)
    min_value = traj.min(axis=0)
    interval = max_value - min_value
    interval[interval == 0] = 1
    traj = (traj - min_value) / interval
    plt.imshow(traj)
    plt.show()


def check_weight(network: torch.nn.Module):
    """
    Check whether network parameters are nan or inf.
    Args:
        network (torch.nn.Module): torch.nn.Module
    Print:
        nan of inf in network params
    """
    for k, v in network.state_dict().items():
        if torch.any(torch.isnan(v)):
            print(k + 'has nan')
        if torch.any(torch.isinf(v)):
            print(k + 'has inf')     


def get_models_parameters(*models):
    """
    return all the parameters of input models in a list
    Args:
        models (torch.nn.Module): all models inputed for getting parameters
    Return:
        list of parameters for all models inputted
    """
    parameters = []
    for model in models:
        parameters += list(model.parameters())
    return parameters


def get_grad_norm(parameters, norm_type: float = 2):
    """
    return all gradient of the parameters
    Args:
        models : parameters of the a model
    Return:
        L2 norm of the gradient
    """
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type) \
                     for p in parameters]), norm_type)
    return total_norm


def get_concat_traj(batch_data: Batch, node_names: List[str]):
    """
    concatenate the data from node_names
    Args:
        batch_data (Batch): Batch of data
        node_names (List): list of node names to get data
    Return:
        data to get
    """
    return torch.cat(get_list_traj(batch_data, node_names), dim=-1)


def get_list_traj(batch_data: Batch, node_names: List[str],
                  nodes_fit_index: dict = None) -> list:
    """
    return all data of node_names from batch_data
    Args:
        batch_data (Batch): Batch of data
        node_names (List): list of node names to get data
        nodes_fit_index (Dict): dict of fixed index for nodel_names
    Return:
        data to get
    """
    datas = []

    for name in node_names:
        if nodes_fit_index:
            datas.append(batch_data[name][..., nodes_fit_index[name]])
        else:
            datas.append(batch_data[name])

    return datas


def generate_rewards(traj: Batch, reward_fn):
    """
    Add rewards for batch trajectories.
    Args:
        traj: batch trajectories.
        reward_fn: how the rewards generate.
    Return:
        batch trajectories with rewards.
    """
    head_shape = traj.shape[:-1]
    traj.reward = reward_fn(traj).view(*head_shape, 1)
    return traj


def generate_rollout(expert_data: Batch,
                     graph: DesicionGraph,
                     traj_length: int,
                     sample_fn=lambda dist: dist.sample(),
                     adapt_stds=None,
                     clip: Union[bool, float] = False,
                     use_target: bool = False):
    """
    Generate trajectories based on current policy.
    Args:
        expert_data: samples from the dataset.
        graph: the computation graph
        traj_length: trajectory length
        sample_fn: sample from a distribution.
    Return:
        batch trajectories.
    NOTE: this function will mantain the last dimension even if it is 1
    """
    assert traj_length <= expert_data.shape[0], \
        'cannot generate trajectory beyond expert data'

    expert_data = deepcopy(expert_data)
    if adapt_stds is None:
        adapt_stds = [None] * (len(graph))

    graph.reset()

    generated_data = []
    current_batch = expert_data[0]

    for i in range(traj_length):
        for node_name, adapt_std in zip(list(graph.keys()), adapt_stds):
            if graph.get_node(node_name).node_type == 'network':
                action_dist = graph.compute_node(node_name, current_batch,
                                                 adapt_std=adapt_std,
                                                 use_target=use_target)
                action = sample_fn(action_dist)
                if isinstance(clip, bool) and clip:
                    action = torch.clamp(action, -1, 1)
                elif isinstance(clip, float):
                    action = torch.clamp(action, -clip, clip)
                else:
                    pass
                current_batch[node_name] = action
                action_log_prob = action_dist.log_prob(action).unsqueeze(dim=-1).detach()
                # TODO: do we need this detach?
                current_batch[node_name + "_log_prob"] = action_log_prob
            else:
                action = graph.compute_node(node_name, current_batch)
                current_batch[node_name] = action

        # check the generated current_batch
        # NOTE: this will make the rollout a bit slower.
        #       Remove it if you are sure no explosion will happend.
        for k, v in current_batch.items():
            if "dist" in k:
                continue
            has_inf = torch.any(torch.isinf(v))
            has_nan = torch.any(torch.isnan(v))
            if has_inf or has_nan:
                logger.warning(f'During rollout detect anomaly data: key {k}, \
                               has inf {has_inf}, has nan {has_nan}')
                logger.warning(f'Should generated rollout with \
                               length {traj_length}, \
                               early stop for only length {i}')
                break

        generated_data.append(current_batch)

        if i == traj_length - 1:
            break
        # clone to new Batch
        current_batch = expert_data[i+1]
        current_batch.update(graph.state_transition(generated_data[-1]))

    generated_data = Batch.stack(generated_data)
    return generated_data


def generate_rollout_bc(expert_data: Batch,
                        graph: DesicionGraph,
                        traj_length: int,
                        sample_fn=lambda dist: dist.sample(),
                        adapt_stds=None,
                        clip: Union[bool, float] = False,
                        use_target: bool = False):
    """
    Generate trajectories based on current policy.
    Args:
        expert_data: samples from the dataset.
        graph: the computation graph
        traj_length: trajectory length
        sample_fn: sample from a distribution.
    Return:
        batch trajectories.
    NOTE: this function will mantain the last dimension even if it is 1
    """

    assert traj_length <= expert_data.shape[0], \
        'cannot generate trajectory beyond expert data'

    expert_data = deepcopy(expert_data)
    if adapt_stds is None:
        adapt_stds = [None] * (len(graph))

    graph.reset()

    generated_data = []
    current_batch = expert_data[0]

    for i in range(traj_length):
        for node_name, adapt_std in zip(list(graph.keys()), adapt_stds):
            if graph.get_node(node_name).node_type == 'network':
                action_dist = graph.compute_node(node_name, current_batch,
                                                 adapt_std=adapt_std,
                                                 use_target=use_target,
                                                 field='bc')
                action = sample_fn(action_dist)
                if isinstance(clip, bool) and clip: 
                    action = torch.clamp(action, -1, 1)
                elif isinstance(clip, float):
                    action = torch.clamp(action, -clip, clip)
                else:
                    pass
                current_batch[node_name] = action
                action_log_prob = \
                    action_dist.log_prob(action).unsqueeze(dim=-1).detach() 
                # TODO: do we need this detach?
                current_batch[node_name + "_log_prob"] = action_log_prob
                current_batch[node_name + "_dist" + f"_{i}"] = action_dist
            else:
                action = graph.compute_node(node_name, current_batch)
                current_batch[node_name] = action

        # check the generated current_batch
        # NOTE: this will make the rollout a bit slower.
        #       Remove it if you are sure no explosion will happend.
        for k, v in current_batch.items():
            if "dist" in k:
                continue
            has_inf = torch.any(torch.isinf(v))
            has_nan = torch.any(torch.isnan(v))
            if has_inf or has_nan:
                logger.warning(f'During rollout detect anomaly data: key {k}, \
                               has inf {has_inf}, has nan {has_nan}')
                logger.warning(f'Should generated rollout with \
                               length {traj_length}, \
                               early stop for only length {i}')
                break

        generated_data.append(current_batch)

        if i == traj_length - 1:
            break
        # clone to new Batch
        current_batch = expert_data[i+1]
        current_batch.update(graph.state_transition(generated_data[-1]))

    generated_data = Batch.stack(generated_data)
    return generated_data


def compute_lambda_return(rewards, values,
                          bootstrap=None, _gamma=0.9, _lambda=0.98):
    """
    Generate lambda return for svg in REVIVE env learning
    Args:
        rewards: reward data for current stated
        values: values derived from value net
        bootstrap: bootstrap for the last time step of next_values
        _gamma: discounted factor
        _lambda: factor for balancing future or current return
    Return:
        discounted return for the input rewards.
    """
    next_values = values[1:]
    if bootstrap is None:
        bootstrap = torch.zeros_like(values[-1])

    next_values = torch.cat([next_values, bootstrap.unsqueeze(0)], dim=0)

    g = [rewards[i] + _gamma * (1 - _lambda) * next_values[i] for i in range(rewards.shape[0])]

    lambda_returns = []
    last = next_values[-1]
    for i in reversed(list(range(len(rewards)))):
        last = g[i] + _gamma * _lambda * last
        lambda_returns.append(last)

    return torch.stack(list(reversed(lambda_returns)))


def sinkhorn_gpu(cuda_id):
    """
    Specifically setting running device
    Args:
        cuda_id: cuda device id
    Return:
        sinkhorn function
    """
    cp.cuda.Device(cuda_id).use()
    import ot.gpu
    return ot.gpu.sinkhorn


def wasserstein_distance(X, Y, cost_matrix,
                         method='sinkhorn', niter=50000, cuda_id=0):
    """
    Calculate wasserstein distance
    Args:
        X & Y : two arrays
        cost_matrix: cost matrix between two arrays
        method: method for calculating w_distance
        niter: number of iteration
        cuda_id: device for calculating w_distance
    Return:
        wasserstein distance
    """
    if method == 'sinkhorn_gpu':
        sinkhorn_fn = sinkhorn_gpu(cuda_id)
        transport_plan = sinkhorn_fn(X, Y, cost_matrix, reg=1, enumItermax=niter)
        # (GPU) Get the transport plan for regularized OT
    elif method == 'sinkhorn':
        transport_plan = ot.sinkhorn(X, Y, cost_matrix, reg=1, numItermax=niter)
        # (CPU) Get the transport plan for regularized OT
    elif method == 'emd':
        transport_plan = ot.emd(X, Y, cost_matrix, numItermax=niter)
        # (CPU) Get the transport plan for OT with no regularisation
    elif method == 'emd2':
        distance = ot.emd2(X, Y, cost_matrix)
        # (CPU) Get the transport loss
        return distance
    else:
        raise NotImplementedError("The method is not implemented!")

    # Calculate Wasserstein by summing diagonals, i.e., W=Trace[MC^T]
    distance = np.sum(np.diag(np.matmul(transport_plan, cost_matrix.T)))

    return distance


def compute_w2_dist_to_expert(policy_trajectorys,
                              expert_trajectorys,
                              scaler=None,
                              data_is_standardscaler=False,
                              max_expert_sampes=20000,
                              dist_metric="euclidean",
                              emd_method="emd",
                              processes=None,
                              use_cuda=False,
                              cuda_id_list=None):
    """
    Computes Wasserstein 2 distance to expert demonstrations.
    Calculate wasserstein distance
    Args:
        policy_trajectorys: data generated by policy
        expert_trajectorys: expert data
        scaler: scale the data
        data_is_standardscaler: whether the data is standard scaled or not
        max_expert_sampes: number of data to use
        dist_metric: distance type,
        emd_method: using cpu for computing
        processes: multi-processing setting
        use_cuda: using gpu for computing
        cuda_id_lis: duda device as list
    Return:
        wasserstein distance

    """
    policy_trajectorys = policy_trajectorys.copy()
    expert_trajectorys = expert_trajectorys.reshape(-1, expert_trajectorys.shape[-1]).copy()
    policy_trajectorys_shape = policy_trajectorys.shape
    policy_trajectorys = policy_trajectorys.reshape(-1, policy_trajectorys_shape[-1])

    expert_trajectorys_index = np.arange(expert_trajectorys.shape[0])
    if expert_trajectorys.shape[0] < max_expert_sampes:
        max_expert_sampes = expert_trajectorys.shape[0]

    if not data_is_standardscaler:
        if scaler is None:
            from sklearn import preprocessing
            scaler = preprocessing.StandardScaler()
            scaler.fit(expert_trajectorys)

        policy_trajectorys = scaler.transform(policy_trajectorys)
        expert_trajectorys = scaler.transform(expert_trajectorys)
    policy_trajectorys = policy_trajectorys.reshape(policy_trajectorys_shape)

    expert_trajectory_weights = \
        1./max_expert_sampes * np.ones(max_expert_sampes)
    policy_trajectory_weights = \
        1./policy_trajectorys_shape[1] * np.ones(policy_trajectorys_shape[1])

    # fallback to cpu mode
    if not CUPY_READY: emd_method = 'emd'

    w2_dist_list = []
    if use_cuda and "gpu" in emd_method:
        if cuda_id_list is None:
            cuda_id_list = list(range(cp.cuda.runtime.getDeviceCount()))
        assert len(cuda_id_list) > 0

        for i, policy_trajectory in enumerate(policy_trajectorys):
            cuda_id = cuda_id_list[i%len(cuda_id_list)]
            cost_matrix = ot.dist(policy_trajectory, expert_trajectorys[expert_trajectorys_index[:max_expert_sampes]], metric=dist_metric)
            w2_dist_list.append(wasserstein_distance(policy_trajectory_weights, expert_trajectory_weights, cost_matrix, emd_method, cuda_id))
    else:
        pool = multiprocessing.Pool(processes = processes if processes is not None else multiprocessing.cpu_count())
        for policy_trajectory in policy_trajectorys:
            cost_matrix = ot.dist(policy_trajectory, expert_trajectorys[expert_trajectorys_index[:max_expert_sampes]], metric=dist_metric)
            np.random.shuffle(expert_trajectorys_index)
            w2_dist_list.append(pool.apply_async(wasserstein_distance, (policy_trajectory_weights, expert_trajectory_weights, cost_matrix, emd_method)))
        pool.close()
        pool.join()
        w2_dist_list = [res.get() for res in w2_dist_list]

    return np.mean(w2_dist_list)


def dict2parser(config: dict):
    """
    transform dict as operation setting as parser
    Args:
        config: dict of operation
    Return:
    parser as command
    """

    parser = argparse.ArgumentParser()

    def get_type(value):
        if type(value) is bool:
            return lambda x: [False, True][int(x)]
        return type(value)

    for k, v in config.items():
        parser.add_argument(f'--{k}', type=get_type(v), default=v)

    return parser


def list2parser(config: List[Dict]):
    """
    transform list of dict as operation setting as parser
    Args:
        config: list of operation
    Return:
    parser as command
    """
    parser = argparse.ArgumentParser()

    def get_type(type_name):
        type_name = eval(type_name) if isinstance(type_name, str) else type_name
        if type_name is bool:
            return lambda x: [False, True][int(x)]
        return type_name

    for d in config:
        names = ['--' + d['name']]
        data_type = get_type(d['type'])
        default_value = d['default']
        addition_args = {}
        if data_type is list:
            data_type = get_type(type(default_value[0])) if type(default_value) is list else get_type(type(default_value))
            addition_args['nargs'] = '+'
        if 'abbreviation' in d.keys():
            names.append('-' + d['abbreviation'])
        parser.add_argument(*names, type=data_type, default=default_value,
                            help=d.get('description', ''), **addition_args)

    return parser


def set_parameter_value(config: List[Dict],
                        name: str,
                        value: Any):
    """
    change value of the name in config file
    Args:
        config: list of dict of variables
        name: the value of the keys to be changed
        value: the value to be chanbed into
    Return:
    resetting default values to the original config
    """
    for param in config:
        if param['name'] == name:
            param['default'] = value
            break
    return config


def update_description(default_description, custom_description):
    '''
    update in-place the default description with a custom description.
    Args:
        default_description:
        custom_description:
    Return:

    '''
    names_to_indexes = {description['name']: i for i, description in enumerate(default_description)}
    for description in custom_description:
        name = description['name']
        index = names_to_indexes.get(name, None)
        if index is None:
            warnings.warn(f'parameter name `{name}` \
                          is not in the default description, skip.')
        else:
            default_description[index] = description 


def find_later(path: str, 
               keyword: str) -> List[str]:
    '''
    find all the later folder after the given keyword
    Args:
        path: a file path to get list of folder
        keyword: the name of the folder which as the last folder at the path
    Return:
        a list of folder as path
    '''
    later = []
    while len(path) > 0:
        path, word = os.path.split(path)
        later.append(word)
        if keyword == word:
            break
    return list(reversed(later))


def get_node_dim_from_dist_configs(dist_configs: dict, node_name: str):
    """
    return the total number of dims of the node_name
    Args:
        dist_configs (dict): decision flow with user setting nodes
        node_name (str): name of the node to get total dimensions
    Return:
        total number of dimensions of the node_name
    """
    node_dim = 0
    for dist_config in dist_configs[node_name]:
        node_dim += dist_config["dim"]

    return node_dim


def save_histogram(histogram_path: str,
                   graph: DesicionGraph,
                   data_loader: DataLoader,
                   device: str,
                   scope: str):
    """
    save the histogram
    Args:
        histogram_path (str): the path to save histogram
        graph (DesicionGraph): DesicionGraph
        data_loader (DataLoader): torch data loader
        device (str): generate data on which device
        scope (str): 'train' or 'val' related to the file-saving name.
    Return:
        Saving the histogram as png file to the histogram_path
    """
    processor = graph.processor

    expert_data = []
    generated_data = []
    for expert_batch in iter(data_loader):
        traj_length = expert_batch.shape[0]
        expert_batch.to_torch(device=device)
        generated_batch = generate_rollout(expert_batch, graph, traj_length, lambda dist: dist.mode, clip=True)
        expert_batch.to_numpy()
        generated_batch.to_numpy()
        expert_data.append(expert_batch)
        generated_data.append(generated_batch)

    # expert_data = {node_name : np.concatenate([batch[node_name] for batch in expert_data], axis=1)  for node_name in graph.keys()}
    expert_data_tmp_dict = {}
    for node_name in graph.keys():
        try:
            expert_data_tmp_dict[node_name] = np.concatenate([batch[node_name] for batch in expert_data], axis=1)
        except Exception as e:
            logger.warning(f"{e} not in dataset when plotting histogram")
            continue

    expert_data = expert_data_tmp_dict
    generated_data = {node_name: np.concatenate([batch[node_name] for batch in generated_data], axis=1)  for node_name in graph.keys()}

    expert_data = processor.deprocess(expert_data)
    generated_data = processor.deprocess(generated_data)

    fig = plt.figure(figsize=(15, 7), dpi=150)
    for node_name in graph.keys():
        index_name = node_name[5:] if node_name in graph.transition_map.values() else node_name
        for i, dimension in enumerate(graph.descriptions[index_name]):
            dimension_name = list(dimension.keys())[0]
            generated_dimension_data = generated_data[node_name][..., i].reshape((-1))
            try:
                expert_dimension_data = expert_data[node_name][..., i].reshape((-1))
                assert expert_dimension_data.shape == generated_dimension_data.shape
            except:
                expert_dimension_data = None

            if dimension[dimension_name]['type'] == 'continuous':
                bins = 100
            elif dimension[dimension_name]['type'] == 'discrete':
                bins = min(dimension[dimension_name]['num'], 100)
            else:
                bins = None

            title = f'{node_name}.{dimension_name}'
            if expert_dimension_data is not None:
                plt.hist([expert_dimension_data, generated_dimension_data], bins=bins, label=['History_Data', 'Generated_Data'], log=True)
            else:
                plt.hist(generated_dimension_data, bins=bins, label='Generated_Data', log=True)
            plt.legend(loc='upper left')
            plt.xlabel(title)
            plt.ylabel("frequency")
            plt.title(title + f"-histogram-{scope}")
            plt.savefig(os.path.join(histogram_path, title + f"-{scope}.png"))
            fig.clf()

    plt.close(fig)


def save_histogram_after_stop(traj_length: int, traj_dir: str, train_dataset, val_dataset):

    """
    save the histogram after the training is stopped
    Args:
        traj_length (int): length of the horizon
        traj_dir (str): saving derectory
        train_dataset: torch data loader
        val_dataset: generate data on which device
    Return:
        Saving the histogram as png file to the histogram_path
    """

    histogram_path = os.path.join(traj_dir, 'histogram')
    if not os.path.exists(histogram_path):
        os.makedirs(histogram_path)

    from revive.data.dataset import collect_data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_graph = torch.load(os.path.join(traj_dir, 'venv_train.pt'), map_location=device).graph
    train_dataset = train_dataset.trajectory_mode_(traj_length)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=False,
                                               collate_fn=partial(collect_data, graph=train_dataset.graph), pin_memory=True)
    save_histogram(histogram_path, train_graph, train_loader, device=device, scope='train')

    val_graph = torch.load(os.path.join(traj_dir, 'venv_val.pt'), map_location=device).graph
    val_dataset = val_dataset.trajectory_mode_(traj_length)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=False,
                                             collate_fn=partial(collect_data, graph=val_dataset.graph), pin_memory=True)
    save_histogram(histogram_path, val_graph, val_loader, device=device, scope='val')


def tb_data_parse(tensorboard_log_dir: str, keys: list = []):
    """
    parse data from tensorboard logdir
    Args:
        tensorboard_log_dir (str): length of the horizon
        keys (list): list of keys to get from tb logdir
    Return:
        geting a dict of result including value of keys
    """

    from tensorboard.backend.event_processing import event_accumulator
    ea = event_accumulator.EventAccumulator(tensorboard_log_dir)
    ea.Reload()
    ea_keys = ea.scalars.Keys()

    ea_keys = [k[9:] if k.startswith('ray/tune/') else k for k in ea_keys]
    parse_data = lambda key: [(i.step, i.value) for i in ea.scalars.Items(key)]

    if keys:
        if set(keys) < set(ea_keys):
            logger.info(f"Keys Error: there are some keys not in tensorboard logs!")
        res = {key: parse_data(key) for key in keys}
    else:
        res = {key: parse_data(key) for key in ea_keys}

    return res


def double_venv_validation(reward_logs, data_reward={}, img_save_path=""):
    """
    policy double venv validation to the img path
    Args:
        reward_logs (str): dict of different rewards
        data_reward (dict): dataset mean reward of train and val dataset
        img_save_path (str): path of saving img
    Return:
        saving double venv validation img to the setting path
    """
    reward_trainPolicy_on_trainEnv = np.array(reward_logs["reward_trainPolicy_on_trainEnv"])
    reward_valPolicy_on_trainEnv = np.array(reward_logs["reward_valPolicy_on_trainEnv"])
    reward_valPolicy_on_valEnv = np.array(reward_logs["reward_valPolicy_on_valEnv"])
    reward_trainPolicy_on_valEnv = np.array(reward_logs["reward_trainPolicy_on_valEnv"])

    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle("Double Venv Validation", fontsize=26)

    x = np.arange(reward_trainPolicy_on_trainEnv[:, 0].shape[0])

    axs[0].plot(x,reward_trainPolicy_on_trainEnv[:, 1], 'r--', label='reward_trainPolicy_on_trainEnv')
    axs[0].plot(x,reward_valPolicy_on_trainEnv[:, 1], 'g--', label='reward_valPolicy_on_trainEnv')
    if "reward_train" in data_reward.keys():
        axs[0].plot(x, np.ones_like(reward_trainPolicy_on_trainEnv[:, 0]) * data_reward["reward_train"], 'b--', label='reward_train')

    axs[0].set_ylabel('Reward')
    axs[0].set_xlabel('Epoch')
    axs[0].legend()

    axs[1].plot(np.arange(reward_valPolicy_on_valEnv[:, 1].shape[0]), reward_valPolicy_on_valEnv[:,1], 'r--', label='reward_valPolicy_on_valEnv')
    axs[1].plot(np.arange(reward_trainPolicy_on_valEnv[:, 1].shape[0]), reward_trainPolicy_on_valEnv[:,1], 'g--', label='reward_trainPolicy_on_valEnv')
    if "reward_val" in data_reward.keys():
        axs[1].plot(np.arange(reward_valPolicy_on_valEnv[:,1].shape[0]), np.ones_like(reward_valPolicy_on_valEnv[:, 0]) * data_reward["reward_val"], 'b--', label='reward_val')

    axs[1].set_ylabel('Reward')
    axs[1].set_xlabel('Epoch')
    axs[1].legend()

    fig.savefig(img_save_path)
    plt.close(fig)


def plt_double_venv_validation(tensorboard_log_dir, reward_train, reward_val, img_save_path):
    """
    Drawing double_venv_validation images
    Args:
        tensorboard_log_dir (str): path of tb infomation
        reward_train : dataset mean reward of train dataset
        reward_val: dataset mean reward of val dataset
        img_save_path (str): path of saving img
    Return:
        saving double venv validation img to the setting path
    """
    reward_logs = tb_data_parse(tensorboard_log_dir, ['reward_trainPolicy_on_valEnv', 'reward_trainPolicy_on_trainEnv',
                                                      'reward_valPolicy_on_trainEnv', 'reward_valPolicy_on_valEnv'])

    data_reward = {"reward_train": reward_train, "reward_val": reward_val}
    double_venv_validation(reward_logs, data_reward, img_save_path)


def _plt_node_rollout(expert_datas, generated_datas, node_name, data_dims, img_save_dir):
    """
    Drawing rollout plot for every node
    Args:
        expert_datas: expert data
        generated_datas: generate data
        node_name: name of the graph node
        data_dims: dimensions of the data
        img_save_path (str): path of saving img
    Return:
        saving double venv validation img to the setting path
    """
    sub_fig_num = len(data_dims)
    if expert_datas is not None:
        for trj_index, (expert_data, generated_data) in enumerate(zip(expert_datas, generated_datas)):
            img_save_path = os.path.join(img_save_dir, f"{trj_index}_{node_name}")
            if sub_fig_num > 1:
                fig, axs = plt.subplots(sub_fig_num, 1, figsize=(15, 5 * sub_fig_num))
                fig.suptitle("Policy Rollout", fontsize=26)
                
                for index, dim in enumerate(data_dims):
                    axs[index].plot(expert_data[:, index], 'r--', label='History Expert Data')
                    axs[index].plot(generated_data[:, index], 'g--', label='Policy Rollout Data')
                    axs[index].set_ylabel(dim)
                    axs[index].set_xlabel('Step')
                    axs[index].legend()
                fig.savefig(img_save_path)
                plt.close(fig)
            else:
                fig = plt.figure(figsize=(15, 5))
                plt.plot(expert_data, 'r--', label='History Expert Data')
                plt.plot(generated_data, 'g--', label='Policy Rollout Data')

                plt.ylabel(data_dims[0])
                plt.xlabel('Step')
                plt.title("Policy Rollout")
                plt.legend()
                plt.savefig(img_save_path)
                plt.close(fig)
    else:
        for trj_index, generated_data in enumerate(generated_datas):
            img_save_path = os.path.join(img_save_dir, f"{trj_index}_{node_name}")
            if sub_fig_num > 1:
                fig, axs = plt.subplots(sub_fig_num, 1, figsize=(15, 5*sub_fig_num))
                fig.suptitle("Policy Rollout", fontsize=26)
                for index,dim in enumerate(data_dims):
                    axs[index].plot(generated_data[:, index], 'g--', label='Policy Rollout Data')

                    axs[index].set_ylabel(dim)
                    axs[index].set_xlabel('Step')
                    axs[index].legend()
                fig.savefig(img_save_path)
                plt.close(fig)
            else:
                fig = plt.figure(figsize=(15, 5))
                plt.plot(generated_data, 'g--', label='Policy Rollout Data')

                plt.ylabel(data_dims[0])
                plt.xlabel('Step')
                plt.title("Policy Rollout")
                plt.legend()
                plt.savefig(img_save_path)
                plt.close(fig)


def save_rollout_action(rollout_save_path: str,
                        graph: DesicionGraph,
                        device: str,
                        dataset,
                        nodes_map,
                        horizion_num = 10):
    """
    save the Trj rollout
    Args:
        rollout_save_path: path of saving img data
        graph: decision graph
        device: device
        dataset: dimensions of the data
        nodes_map: graph nodes
        horizion_num (int): length to generate data
    Return:
        save Trj rollout
    """
    if not os.path.exists(rollout_save_path):
        os.makedirs(rollout_save_path)

    graph = graph.to(device)

    from revive.data.dataset import collect_data

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=horizion_num,
                                              shuffle=True,
                                              collate_fn=partial(collect_data, graph=graph),
                                              pin_memory=True)

    processor = graph.processor
    expert_data = []
    generated_data = []

    for expert_batch in iter(data_loader):
        traj_length = expert_batch.shape[0]
        expert_batch.to_torch(device=device)
        generated_batch = generate_rollout(expert_batch, graph, traj_length, lambda dist: dist.mode, clip=True)
        expert_batch.to_numpy()
        generated_batch.to_numpy()
        expert_data.append(expert_batch)
        generated_data.append(generated_batch)
        break

    # expert_data = {node_name : np.concatenate([batch[node_name] for batch in expert_data if node_name in batch.keys()], axis=1)  for node_name in nodes_map.keys()}
    expert_data_tmp_dict = {}
    for node_name in nodes_map.keys():
        try:
            expert_data_tmp_dict[node_name] = np.concatenate([batch[node_name] for batch in expert_data], axis=1)
        except Exception as e:
            logger.warning(f"{e} not in dataset when plotting rollout")
            continue
    expert_data = expert_data_tmp_dict
    generated_data = {node_name: np.concatenate([batch[node_name] for batch in generated_data], axis=1) for node_name in nodes_map.keys()}

    # deprocess all data
    expert_data = processor.deprocess(expert_data)
    generated_data = processor.deprocess(generated_data)

    # del ts_node in nodes_map
    if dataset.ts_node_frames:
        ts_nodes_map = {}
        for k, v in dataset.ts_node_frames.items():
            ts_nodes_map[k] = nodes_map[k][-len(nodes_map[k])//v:]
            if 'next_' + k in nodes_map.keys():
                ts_nodes_map['next_' + k] = nodes_map['next_' + k][-len(nodes_map['next_' + k])//v:]
        nodes_map.update(ts_nodes_map)

    if graph.ts_nodes:
        for ts_node,node in graph.ts_nodes.items():
            if ts_node in nodes_map.keys():
                if node in nodes_map.keys():
                    expert_data.pop(ts_node)
                    generated_data.pop(ts_node)
                    nodes_map.pop(ts_node)
                else:
                    nodes_map[node] = [c[c.index("_")+1:]for c in nodes_map[ts_node]]
                    expert_data[node] = expert_data[ts_node][...,-len(nodes_map[node]):]
                    generated_data[node] = generated_data[ts_node][...,-len(nodes_map[node]):]
                    nodes_map.pop(ts_node)

            if "next_" + ts_node in nodes_map.keys():
                if "next_" + node in nodes_map.keys():
                    expert_data.pop("next_" + ts_node)
                    generated_data.pop("next_" + ts_node)
                    nodes_map.pop("next_" + ts_node)
                else:
                    nodes_map["next_" + node] = [c[c.index("_")+1:]for c in nodes_map["next_" + ts_node]]
                    expert_data["next_" + node] = expert_data["next_" +ts_node][..., -len(nodes_map["next_" + node]):]
                    generated_data["next_" + node] = generated_data["next_" +ts_node][..., -len(nodes_map["next_" + node]):]
                    nodes_map.pop("next_" + ts_node)

    #select_indexs = np.random.choice(np.arange(expert_data[list(nodes.keys())[0]].shape[1]), size=10, replace=False)

    horizion_num = min(horizion_num, expert_data.shape[1])
    for node_name, node_dims in nodes_map.items():
        if node_name in ["step_node_", "traj"]:
            continue
        try:
            expert_action_data = expert_data[node_name]
            select_expert_action_data = [expert_action_data[:,index] for index in range(horizion_num)]
        except Exception as e:
            logger.warning(f"{e} not in dataset when plotting rollout")
            select_expert_action_data = None

        generated_action_data = generated_data[node_name]
        select_generated_action_data = [generated_action_data[: , index] for index in range(horizion_num)]

        node_rollout_save_path = os.path.join(rollout_save_path, node_name)
        if not os.path.exists(node_rollout_save_path):
            os.makedirs(node_rollout_save_path)

        _plt_node_rollout(select_expert_action_data, select_generated_action_data, node_name, node_dims, node_rollout_save_path)


def data_to_dtreeviz(data: pd.DataFrame,
                     target: pd.DataFrame,
                     target_type: (List[str], str),
                     orientation: ('TD', 'LR') = "TD",
                     fancy: bool = True,
                     max_depth: int = 3,
                     output: (str) = None):
    """
    pd data to decision tree
    Args:
        data: dataset in pandas form
        target: target in pandas form
        target_type: continuous or discrete
        orientation: Left to right or top to down
        fancy: true or false for dtreeviz function
        max_depth (int): depth of the tree
        output: whether to output dtreeviz result in the path
    Return:
        save Trj rollout
    """
    # orange = '#F46d43'
    # red = '#FF0018'
    # blue ='#0000F9'
    # black = '#000000'
    #
    custom_color = {'scatter_edge': '#225eab',         
                    'scatter_marker': '#225eab',
                    'scatter_marker_alpha':0.3,

                    'wedge': '#F46d43', #orange
                    'split_line': '#F46d43', #orange
                    'mean_line': '#F46d43', #orange
                    
                    'axis_label': '#000000', #black
                    'title': '#000000', #black,
                    'legend_title': '#000000', #black,
                    'legend_edge': '#000000', #black,
                    'edge': '#000000', #black,
                    'color_map_min': '#c7e9b4',
                    'color_map_max': '#081d58',
                    # 'classes': color_blind_friendly_colors,
                    'rect_edge': '#000000', #black,
                    'text': '#000000', #black,
                    # 'highlight': 'k',

                    'text_wedge': '#000000', #black,
                    'arrow': '#000000', #black,
                    'node_label': '#000000', #black,
                    'tick_label': '#000000', #black,
                    'leaf_label': '#000000', #black,
                    'pie': '#000000', #black,      
                    }


    if isinstance(target_type, str) and len(target.columns) > 1:
        target_type = [target_type, ] * len(target.columns)

    _tmp_pdf_paths = []

    with TemporaryDirectory() as dirname:
        for _target_type, target_name in zip(target_type,target.columns):
            if _target_type == "Classification" or _target_type == "C":
                _target_type = "Classification"
                decisiontree = tree.DecisionTreeClassifier(max_depth=max_depth)
            elif _target_type == "Regression" or _target_type == "R":
                _target_type = "Regression"
                decisiontree = tree.DecisionTreeRegressor(max_depth=max_depth, random_state=1)
            else:
                raise NotImplementedError

            _target = target[[target_name, ]]
            decisiontree.fit(data, _target)

            temp_size = min(data.values.shape[0], 5000)
            np.random.seed(2020)
            random_index = np.random.choice(np.arange(data.values.shape[0]), size=(temp_size,))


            if _target_type == "Classification":
                _orientation = "TD"
                class_names = list(set(list(_target.values.reshape(-1))))
                _target = _target.values[random_index,:].reshape(-1)

            else:
                _target = _target.values[random_index,:]
                class_names = None
                _orientation = "LR"

            viz = dtreeviz(decisiontree,
                           data.values[random_index,:],
                           _target,
                            target_name=target_name,
                            feature_names=data.columns,
                            class_names=class_names,

                            show_root_edge_labels = True,
                            show_node_labels = False,

                            title = _target_type + " Decision Tree of "+ '<'+ target_name + '>',
                            orientation = _orientation,

                            label_fontsize = 15,
                            ticks_fontsize = 10,
                            title_fontsize = 15,

                            fancy=fancy,
                            scale = 2,
                            colors = custom_color,
                            cmap = "cool",)

            _tmp_pdf_path = os.path.join(dirname, str(uuid1())+".pdf")
            _tmp_pdf_paths.append(_tmp_pdf_path)

            try:
                svg2pdf(url=viz.save_svg(), 
                        output_width=1000, 
                        output_height=1000, 
                        write_to=_tmp_pdf_path)
            except Exception as e:
                logger.error(f"{e}")
                try:                
                    os.system('sudo apt install graphviz')
                    svg2pdf(url=viz.save_svg(), 
                            output_width=1000, 
                            output_height=1000, 
                            write_to=_tmp_pdf_path)
                except Exception as e:
                    logger.error(f"{e}")
                    logger.info(f"if Graphviz show Error, please try to use <sudo apt install graphviz> in operation system")
                    return

        if output is None:
            if len(vizs.keys()) == 1:
                return viz
            return vizs

        assert output, f"output should be not None"

        if len(_tmp_pdf_paths)==len(target_type):
            merger = PdfFileMerger()

            for in_pdf in _tmp_pdf_paths:
                with open(in_pdf, 'rb') as pdf:
                    merger.append(PdfFileReader(pdf))
            merger.write(output)

        return


def net_to_tree(tree_save_path: str,
                graph: DesicionGraph,
                device: str,
                dataset,
                nodes):
    """
    deriving the net model to decision tree
    Args:
        tree_save_path: result saving path
        graph: decision flow in DesicionGraph type
        device: device to generate data
        dataset: dataset for deriving decision tree
        nodes: nodes in decision flow to derive decision tree
    Return:
        save decision tree
    """
    if not os.path.exists(tree_save_path):
        os.makedirs(tree_save_path)

    graph = graph.to(device)

    from revive.data.dataset import collect_data

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=256,
                                              shuffle=True,
                                              collate_fn=partial(collect_data, graph=graph),
                                              pin_memory=True)

    processor = graph.processor
    expert_data = []
    generated_data = []
    data_num = 0
    for expert_batch in iter(data_loader):
        traj_length = expert_batch.shape[0]
        expert_batch.to_torch(device=device)
        generated_batch = generate_rollout(expert_batch, graph, traj_length, lambda dist: dist.mode, clip=True)
        expert_batch.to_numpy()
        generated_batch.to_numpy()
        expert_data.append(expert_batch)
        generated_data.append(generated_batch)

        data_num += traj_length*256
        if data_num > 20000:
            break
    data_keys = list(graph.keys()) + graph.leaf
    # expert_data = {node_name : np.concatenate([batch[node_name] for batch in expert_data], axis=1)  for node_name in data_keys}
    expert_data_tmp_dict = {}
    for node_name in data_keys:
        try:
            val = np.concatenate([batch[node_name] for batch in expert_data], axis=1)
            expert_data_tmp_dict[node_name] = val.reshape(-1, val.shape[-1])
        except:
            continue
    expert_data = expert_data_tmp_dict

    generated_data = {node_name: np.concatenate([batch[node_name] for batch in generated_data], axis=1)  for node_name in data_keys}
    generated_data = {node_name: node_data.reshape(-1,node_data.shape[-1]) for node_name, node_data in generated_data.items()}
    expert_data = processor.deprocess(expert_data)
    generated_data = processor.deprocess(generated_data)

    sample_num = expert_data[list(nodes.keys())[0]].shape[0]
    size = min(sample_num, 100000)
    select_indexs = np.random.choice(np.arange(sample_num), size=size, replace=False)

    for output_node, output_node_dims in nodes.items():
        input_node_dims = []
        input_nodes = graph[output_node]
        for input_node in input_nodes:
            for obs_dim in graph.descriptions[input_node]:
                input_node_dims.append(list(obs_dim.keys())[0])

        input_data = np.concatenate([generated_data[node] for node in input_nodes], axis=-1)
        output_data = generated_data[output_node]

        input_data = input_data[select_indexs]
        output_data = output_data[select_indexs]

        X = pd.DataFrame(input_data, columns=input_node_dims) 
        Y = pd.DataFrame(output_data, columns=output_node_dims)

        # begin bug fix tuzuolin 2022-1010 
        # Y_type = "R"
        from collections import ChainMap # tuzuolin 2022 1011
        temp_dict = dict(ChainMap(*graph.descriptions[output_node]))
        Y_type = []
        for i_type in map(lambda x: temp_dict[x]['type'], output_node_dims):
            Y_type.append("C" if i_type == 'category' else "R")

        result = data_to_dtreeviz(X, Y, Y_type, output=os.path.join(tree_save_path,output_node+".pdf"))


def generate_response_inputs(expert_data: Batch, 
                     dataset: Batch,
                     graph: DesicionGraph, 
                     obs_sample_num=16):

    expert_data = deepcopy(expert_data)
    all_input_nodes = set([item for sub_ls in graph.graph_dict.values() for item in sub_ls])
    """
    Output:
        generated_inputs: Dict[str: Dict[tuple: np.ndarray]]
        generated_inputs.keys(): ['obs', 'action', 'door_open', ...] --> the nodes appear in the input of any network node
        generated_inputs['obs'].keys(): [(0, 0), ..., (0, 15), (1, 0), ..., (1, 15), ..., (obs_dim, 0), ..., (obs_dim, 15)] --> (data_dim_index, sub_graph_index)
        generated_inputs['obs'][(0, 0)]: np.ndarray of shape (dim_sample_num, obs_dim) --> perturbated data

        dim_perturbations: Dict[str: Dict[int: np.ndarray]]
        dim_perturbations.keys(): ['obs', 'action', 'door_open', ...] --> the nodes appear in the input of any network node
        dim_perturbations['obs'].keys(): [0, 1, ..., obs_dim]
        dim_perturbations['obs'][0]: np.ndarray of shape (dim_sample_num, ) --> dim_perturbation --> as x-axis in plotting
    """
    generated_inputs = defaultdict(dict)
    dim_perturbations = defaultdict(dict)

    for node_name in all_input_nodes:
        descriptions = graph.descriptions[node_name]
        for dim, description in  enumerate(descriptions):
            # generate fake data
            # info = list(description.values())[0]
            for _, info in description.items():
                if info['type'] == 'continuous':
                    dim_sample_num = 100
                    dim_perturbation = np.linspace(info.get('min') if info.get('min') is not None else dataset[node_name][:, dim].min(), info.get('max') if info.get('max') is not None else dataset[node_name][:, dim].max(), num=dim_sample_num)
                elif info['type'] == 'category':
                    values = info['values']
                    dim_sample_num = len(values)
                    dim_perturbation = np.array(values)
                elif info['type'] == 'discrete':
                    assert isinstance(info, dict), "in discrete, assert isinstance(info, dict)"
                    dim_sample_num = info['num']
                    dim_perturbation = np.linspace(info['min'], info['max'], num=dim_sample_num)
                else:
                    raise NotImplementedError("data dim_type not match")

                for index in range(obs_sample_num):
                    inputs = deepcopy(expert_data[node_name][index])  # (dim, )
                    inputs = np.repeat(np.expand_dims(inputs, axis=0), repeats=dim_sample_num, axis=0)  # (dim_sample_num, dim)
                    inputs[:, dim] = dim_perturbation
                    generated_inputs[node_name][(dim, index)] = inputs
                    dim_perturbations[node_name][dim] = dim_perturbation

    return generated_inputs, dim_perturbations


def generate_response_outputs(generated_inputs: defaultdict, 
                     expert_data: Batch, 
                     venv_train: VirtualEnvDev,
                     venv_val: VirtualEnvDev):

    generated_inputs = deepcopy(generated_inputs)
    """
    Output:
        generated_outputs: Dict[str: Dict[tuple: np.ndarray]]
        generated_outputs.keys(): ['action', 'next_obs', ...] --> the nodes in graph.keys() whose node_type == 'network'
        generated_outputs['next_obs'].keys(): [(0, 0, input_1), ..., (0, 0, input_n), 
                                                    (0, 1, input_1), ..., (0, 1, input_n),
                                                    ...,
                                                    (0, 15, input_1), ..., (0, 15, input_n),
                                            (1, 0, input_1), ..., (1, 0, input_n),
                                                    (1, 1, input_1), ..., (1, 1, input_n),
                                                    ...,
                                                    (1, 15, input_1), ..., (1, 15, input_n),
                                                ...,
                                            (next_obs_dim, 0, input_1), ..., (next_obs_dim, 0, input_n),
                                                    (next_obs_dim, 1, input_1), ..., (next_obs_dim, 1, input_n),
                                                    ...,
                                                    (next_obs_dim, 15, input_1), ..., (next_obs_dim, 15, input_n)] --> (data_dim_index, sub_graph_index, each_input_name)
        generated_outputs['next_obs'][(0, 0, input_1)]: np.ndarray of shape (dim_sample_num, next_obs_dim) --> outputs corresponding to each perturbated input
    """
    graph = venv_train.graph
    generated_outputs_train = defaultdict(dict)
    generated_outputs_val = defaultdict(dict)
    with torch.no_grad():
        for node_name in list(graph.keys()):
            if graph.get_node(node_name).node_type == 'network':
                input_names = graph.get_node(node_name).input_names
                inputs_dict = graph.get_node(node_name).get_inputs(generated_inputs)  # Dict[str: Dict[tuple: np.ndarray]]
                for input_name in input_names:
                    for dim, index in list(inputs_dict[input_name].keys()):
                        state_dict = {}
                        state_dict[input_name] = inputs_dict[input_name][(dim, index)]
                        state_dict.update({name: np.repeat(np.expand_dims(expert_data[name][index], axis=0), repeats=state_dict[input_name].shape[0], axis=0) 
                                                for name in input_names if name != input_name})
                        venv_train.reset()
                        venv_val.reset()
                        generated_outputs_train[node_name][(dim, index, input_name)] = venv_train.node_infer(node_name, state_dict)
                        generated_outputs_val[node_name][(dim, index, input_name)] = venv_val.node_infer(node_name, state_dict)

    return generated_outputs_train, generated_outputs_val


def plot_response_curve(response_curve_path, graph_train, graph_val, dataset, device, obs_sample_num=16):
    if not os.path.exists(response_curve_path):
        os.makedirs(response_curve_path)

    dataset = deepcopy(dataset)

    graph_train.reset()
    graph_train = deepcopy(graph_train)
    venv_train = VirtualEnvDev(graph_train)
    venv_train.to(device)

    graph_val.reset()
    graph_val = deepcopy(graph_val)
    venv_val = VirtualEnvDev(graph_val)
    venv_val.to(device)

    indexes = np.random.choice(np.arange(dataset.shape[0]), size=(obs_sample_num, ), replace=False)
    expert_data = dataset[indexes]  # numpy array

    generated_inputs, dim_perturbations = generate_response_inputs(expert_data, dataset, graph_train, obs_sample_num)
    generated_outputs_train, generated_outputs_val = generate_response_outputs(generated_inputs, expert_data, venv_train, venv_val)

    obs_sample_num_per_dim = int(np.sqrt(obs_sample_num))
    with torch.no_grad():
        for node_name in list(graph_train.keys()):
            if graph_train.get_node(node_name).node_type == 'network':
                input_names = graph_train.get_node(node_name).input_names
                output_dims = dataset[node_name].shape[-1]
                for input_name in input_names:
                    input_dims = dataset[input_name].shape[-1]
                    # plot
                    fig = plt.figure(figsize=(8 * input_dims, 10 * output_dims))  # (width, height)
                    red_patch = mpatches.Patch(color='red', label='venv_train')
                    blue_patch = mpatches.Patch(color='blue', label='venv_val')
                    outer = gridspec.GridSpec(output_dims, input_dims, wspace=0.2, hspace=0.2)
                    for output_dim in range(output_dims):
                        for input_dim in range(input_dims):
                            dim_perturbation = dim_perturbations[input_name][input_dim]
                            outer_index = output_dim * input_dims + input_dim
                            inner = gridspec.GridSpecFromSubplotSpec(obs_sample_num_per_dim, obs_sample_num_per_dim, subplot_spec=outer[outer_index], wspace=0.2, hspace=0.2)
                            outer_ax = plt.Subplot(fig, outer[outer_index])
                            outer_ax.axis('off')
                            fig.add_subplot(outer_ax)
                            mae = []
                            corr = []
                            for index in range(obs_sample_num):
                                ax = plt.Subplot(fig, inner[index])
                                if output_dim == 0 and input_dim == 0 and index == 0:
                                    ax.legend(handles=[red_patch, blue_patch], loc="upper left")
                                output_train = generated_outputs_train[node_name][(input_dim, index, input_name)][:, output_dim]
                                output_val = generated_outputs_val[node_name][(input_dim, index, input_name)][:, output_dim]
                                line1, = ax.plot(dim_perturbation, output_train, color='red')
                                line2, = ax.plot(dim_perturbation, output_val, color='blue')
                                fig.add_subplot(ax)
                                mae.append(np.abs(output_train - output_val) / dim_perturbation.shape[0])
                                corr.append(stats.spearmanr(output_train, output_val).correlation)
                            outer_ax.set_title(f"{node_name}_dim: {output_dim}, {input_name}_dim: {input_dim}\nMAE: {np.sum(mae) / obs_sample_num:.2f}\nCorr: {np.sum(corr) / obs_sample_num:.2f}", )

                    response_curve_node_path = os.path.join(response_curve_path, node_name)
                    if not os.path.exists(response_curve_node_path):
                        os.makedirs(response_curve_node_path)
                    plt.savefig(os.path.join(response_curve_node_path, f"{node_name}_on_{input_name}.png"), bbox_inches='tight')


def response_curve(response_curve_path, venv, dataset, device="cuda" if torch.cuda.is_available() else "cpu", obs_sample_num=16):
    def generate_response_outputs(generated_inputs: defaultdict, expert_data: Batch, venv: VirtualEnvDev):
        generated_inputs = deepcopy(generated_inputs)
        graph = venv.graph
        generated_outputs_train = defaultdict(dict)
        with torch.no_grad():
            for node_name in list(graph.keys()):
                if graph.get_node(node_name).node_type == 'network':
                    input_names = graph.get_node(node_name).input_names
                    inputs_dict = graph.get_node(node_name).get_inputs(generated_inputs)  # Dict[str: Dict[tuple: np.ndarray]]
                    for input_name in input_names:
                        for dim, index in list(inputs_dict[input_name].keys()):
                            state_dict = {}
                            state_dict[input_name] = inputs_dict[input_name][(dim, index)]
                            state_dict.update({name: np.repeat(np.expand_dims(expert_data[name][index], axis=0), repeats=state_dict[input_name].shape[0], axis=0) 
                                                    for name in input_names if name != input_name})
                            venv.reset()
                            generated_outputs_train[node_name][(dim, index, input_name)] = venv.node_infer(node_name, state_dict)
        return generated_outputs_train

    if not os.path.exists(response_curve_path):
        os.makedirs(response_curve_path)

    dataset = deepcopy(dataset)

    venv.to(device)
    graph = venv.graph
    graph.reset()

    indexes = np.random.choice(np.arange(dataset.shape[0]), size=(obs_sample_num, ), replace=False)
    expert_data = dataset[indexes]  # numpy array

    generated_inputs, dim_perturbations = generate_response_inputs(expert_data, dataset, graph, obs_sample_num)
    generated_outputs = generate_response_outputs(generated_inputs, expert_data, venv)

    obs_sample_num_per_dim = int(np.sqrt(obs_sample_num))
    with torch.no_grad():
        for node_name in list(graph.keys()):
            if graph.get_node(node_name).node_type == 'network':
                input_names = graph.get_node(node_name).input_names
                output_dims = dataset[node_name].shape[-1]
                for input_name in input_names:
                    input_dims = dataset[input_name].shape[-1]
                    # plot
                    fig = plt.figure(figsize=(8 * input_dims, 10 * output_dims))  # (width, height)
                    blue_patch = mpatches.Patch(color='blue', label='venv')
                    outer = gridspec.GridSpec(output_dims, input_dims, wspace=0.2, hspace=0.2)
                    for output_dim in range(output_dims):
                        for input_dim in range(input_dims):
                            dim_perturbation = dim_perturbations[input_name][input_dim]
                            outer_index = output_dim * input_dims + input_dim
                            inner = gridspec.GridSpecFromSubplotSpec(obs_sample_num_per_dim, obs_sample_num_per_dim, subplot_spec=outer[outer_index], wspace=0.2, hspace=0.2)
                            outer_ax = plt.Subplot(fig, outer[outer_index])
                            outer_ax.axis('off')
                            fig.add_subplot(outer_ax)
                            for index in range(obs_sample_num):
                                ax = plt.Subplot(fig, inner[index])
                                if output_dim == 0 and input_dim == 0 and index == 0:
                                    ax.legend(handles=[blue_patch, ], loc="upper left")
                                output = generated_outputs[node_name][(input_dim, index, input_name)][:, output_dim]
                                line1, = ax.plot(dim_perturbation, output, color='blue')
                                fig.add_subplot(ax)
                            outer_ax.set_title(f"{node_name}_dim: {output_dim}, {input_name}_dim: {input_dim}", )

                    response_curve_node_path = os.path.join(response_curve_path, node_name)
                    if not os.path.exists(response_curve_node_path):
                        os.makedirs(response_curve_node_path)
                    plt.savefig(os.path.join(response_curve_node_path, f"{node_name}_on_{input_name}.png"), bbox_inches='tight')


def create_unit_vector(d_model):
    """
    Normalization of unit vector
    Args:
        size: tuple, (num_vecs, dim) / (dim, )
    Return:
        a random vector of which the sum eaquals 1
    """
    x = np.random.uniform(low=-1, high=1, size=d_model)
    x -= x.mean()
    return x / np.linalg.norm(x)


def generate_bin_encoding(traj_num):
    """
    transform trajectory id into binary code in form of list
    Args:
        traj_num: the number of traj in data
    Return:
        binary vector of traj id
    """
    traj_str = "{0:b}".format(traj_num)
    max_len = len(traj_str)
    traj_encodings = []
    for i in range(traj_num):
        traj_encodings.append(np.array(list(map(int, list("{0:b}".format(i).zfill(max_len))))))
    return np.vstack(traj_encodings)


class PositionalEncoding:
    def __init__(self, d_model: int, max_len: int = 5000):
        position = np.expand_dims(np.arange(max_len), axis=1)
        div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = np.zeros((max_len, 1, d_model))
        pe[:, 0, 0::2] = np.sin(position * div_term)
        pe[:, 0, 1::2] = np.cos(position * div_term)
        self.pe = pe

    def encode(self, pos) -> np.ndarray:
        """
        Args:
            pos: Array, shape [seq_len, 1] / scalar
        Output:
            pe: Array, shape [seq_len, d_model] / [d_model, ]
        """
        pe = self.pe[pos].squeeze()  # [seq_len, d_model]
        return pe