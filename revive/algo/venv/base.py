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

import os 
import ray
import shutil
import torch
import pickle
import warnings
import importlib
import traceback
import numpy as np
from scipy import stats
from loguru import logger
from copy import deepcopy
from ray import tune
from ray import train
from ray.train.torch import TorchTrainer
from revive.computation.graph import DesicionGraph
from revive.computation.inference import *
from revive.data.batch import Batch
from revive.data.dataset import data_creator
from revive.utils.raysgd_utils import NUM_SAMPLES, AverageMeterCollection
from revive.utils.tune_utils import get_tune_callbacks, CustomSearchGenerator, CustomBasicVariantGenerator, CLIReporter
from revive.utils.common_utils import *
from revive.utils.auth_utils import customer_uploadTrainLog

warnings.filterwarnings('ignore')

def catch_error(func):
    '''push the training error message to data buffer'''
    def wrapped_func(self, *args, **kwargs):
        return func(self, *args, **kwargs)
        """
        if True:
            return func(self, *args, **kwargs)
        else:
            error_message = traceback.format_exc()
            logger.warning('Detect error:{}, Error Message: {}'.format(e,error_message))
            ray.get(self._data_buffer.update_status.remote(self._traj_id, 'error', error_message))
            self._stop_flag = True
            try:
                customer_uploadTrainLog(self.config["trainId"],
                                        os.path.join(os.path.abspath(self._workspace),"revive.log"),
                                        "train.simulator",
                                        "fail",
                                        self._acc,
                                        self.config["accessToken"])
            except Exception as e:
                logger.info(f"{e}")
            return {
                'stop_flag' : True,
                'now_metric' : np.inf,
                'least_metric' : np.inf,
            }
        """
    return wrapped_func

class VenvOperator():
    r"""
    The base venv class.
    """
    NAME = None # this need to be set in any subclass
    r"""
    Name of the used algorithm.
    """

    @property
    def metric_name(self): 
        r"""
        This define the metric we try to minimize with hyperparameter search.
        """
        return f"{self.NAME}/average_{self.config['venv_metric']}"

    @property
    def nodes_models_train(self):
        return self.train_models[:self.config['learning_nodes_num']]

    @property
    def other_models_train(self):
        return self.train_models[self.config['learning_nodes_num']:]

    @property
    def nodes_models_val(self):
        return self.val_models[:self.config['learning_nodes_num']]

    @property
    def other_models_val(self):
        return self.val_models[self.config['learning_nodes_num']:]

    # NOTE: you need either write the `PARAMETER_DESCRIPTION` or overwrite `get_parameters` and `get_tune_parameters`.
    PARAMETER_DESCRIPTION = [] # a list of dict to describe the parameter of the algorithm

    @classmethod
    def get_parameters(cls, command=None, **kargs):
        parser = list2parser(cls.PARAMETER_DESCRIPTION)
        return parser.parse_known_args(command)[0].__dict__

    @classmethod
    def get_tune_parameters(cls, config : dict, **kargs):
        r"""
        Use ray.tune to wrap the parameters to be searched.
        """
        _search_algo = config['venv_search_algo'].lower()

        tune_params = {
            "name": "venv_tune",
            "reuse_actors": config["reuse_actors"],
            "local_dir": config["workspace"],
            "callbacks": get_tune_callbacks(),
            "stop": {
                "stop_flag": True
            },
            "verbose": config["verbose"],
        }

        if _search_algo == 'random':
            random_search_config = {}

            for description in cls.PARAMETER_DESCRIPTION:
                if 'tune' in description.keys() and not description["tune"]:
                    continue 
                if 'search_mode' in description.keys():
                    if description['search_mode'] == 'continuous': 
                        random_search_config[description['name']] = tune.uniform(*description['search_values'])
                    elif description['search_mode'] == 'grid' or description['search_mode'] == 'discrete':
                        random_search_config[description['name']] = tune.choice(description['search_values'])

            config["total_num_of_trials"] = config['train_venv_trials']
            tune_params['config'] = random_search_config
            tune_params['num_samples'] = config["total_num_of_trials"]
            tune_params['search_alg'] = CustomBasicVariantGenerator()

        elif _search_algo == 'zoopt':
            # from ray.tune.search.zoopt import ZOOptSearch
            from revive.utils.tune_utils import ZOOptSearch
            from zoopt import ValueType

            if config['parallel_num'] == 'auto':
                if config['use_gpu']:
                    num_of_gpu = int(ray.available_resources()['GPU'])
                    num_of_cpu = int(ray.available_resources()['CPU'])
                    parallel_num = min(int(num_of_gpu / config['venv_gpus_per_worker']), num_of_cpu)
                else:
                    num_of_cpu = int(ray.available_resources()['CPU'])
                    parallel_num = num_of_cpu
            else:
                parallel_num = int(config['parallel_num'])

            assert parallel_num > 0
            config['parallel_num'] = parallel_num

            dim_dict = {}

            for description in cls.PARAMETER_DESCRIPTION:
                if 'tune' in description.keys() and not description["tune"]:
                    continue 
                if 'search_mode' in description.keys():
                    if description['search_mode'] == 'continuous': 
                        dim_dict[description['name']] = (ValueType.CONTINUOUS, description['search_values'], min(description['search_values']))
                    elif description['search_mode'] == 'discrete':
                        dim_dict[description['name']] = (ValueType.DISCRETE, description['search_values'])
                    elif description['search_mode'] == 'grid':
                        dim_dict[description['name']] = (ValueType.GRID, description['search_values'])

            zoopt_search_config = {
                "parallel_num": config['parallel_num']
            }

            config["total_num_of_trials"] = config['train_venv_trials']
            tune_params['search_alg'] = ZOOptSearch(
                algo="Asracos",  # only support Asracos currently
                budget=config["total_num_of_trials"],
                dim_dict=dim_dict,
                metric='least_metric',
                mode="min",
                **zoopt_search_config
            )
            tune_params['config'] = dim_dict
            tune_params['search_alg'] = CustomSearchGenerator(tune_params['search_alg'])  # wrap with our generator
            tune_params['num_samples'] = config["total_num_of_trials"]

        elif 'grid' in _search_algo:
            grid_search_config = {}
            config["total_num_of_trials"] = 1

            for description in cls.PARAMETER_DESCRIPTION:
                if 'tune' in description.keys() and not description["tune"]:
                    continue 
                if 'search_mode' in description.keys():
                    if description['search_mode'] == 'grid': 
                        grid_search_config[description['name']] = tune.grid_search(description['search_values'])
                        config["total_num_of_trials"] *= len(description['search_values'])
                    else:
                        warnings.warn(f"Detect parameter {description['name']} is define as searchable in `PARAMETER_DESCRIPTION`. " + \
                            f"However, since grid search does not support search type {description['search_mode']}, the parameter is skipped. " + \
                            f"If this parameter is important to the performance, you should consider other search algorithms. ")

            tune_params['config'] = grid_search_config
            tune_params['num_samples'] = config["total_num_of_trials"]
            tune_params['search_alg'] = CustomBasicVariantGenerator()

        elif 'bayes' in _search_algo:
            from ray.tune.search.bayesopt import BayesOptSearch
            bayes_search_config = {}

            for description in cls.PARAMETER_DESCRIPTION:
                if 'tune' in description.keys() and not description["tune"]:
                    continue 
                if 'search_mode' in description.keys():
                    if description['search_mode'] == 'continuous': 
                        bayes_search_config[description['name']] = description['search_values']
                    else:
                        warnings.warn(f"Detect parameter {description['name']} is define as searchable in `PARAMETER_DESCRIPTION`. " + \
                            f"However, since bayesian search does not support search type {description['search_mode']}, the parameter is skipped. " + \
                            f"If this parameter is important to the performance, you should consider other search algorithms. ")
            
            config["total_num_of_trials"] = config['train_venv_trials']
            
            tune_params['config'] = bayes_search_config
            tune_params['search_alg'] = BayesOptSearch(bayes_search_config, metric="least_metric", mode="min") 
            tune_params['search_alg'] = CustomSearchGenerator(tune_params['search_alg'])  # wrap with our generator
            tune_params['num_samples'] = config["total_num_of_trials"]
        
        else:
            raise ValueError(f'search algorithm {_search_algo} is not supported!')
        
        reporter = CLIReporter(parameter_columns=list(tune_params['config'].keys()), max_progress_rows=50, max_report_frequency=10, sort_by_metric=True)
        reporter.add_metric_column("least_metric", representation='least metric loss')
        reporter.add_metric_column("now_metric", representation='current metric loss')
        
        tune_params["progress_reporter"] = reporter

        return tune_params

    def model_creator(self, config : dict, graph : DesicionGraph):
        r"""
        Create all the models. The algorithm needs to define models for the nodes to be learned.

        Args:
            :config: configuration parameters
        
        Return: 
            a list of models

        """
        raise NotImplementedError

    def optimizer_creator(self, models : List[torch.nn.Module], config : dict):
        r"""
        Define optimizers for the created models.

        Args:
            :pmodels: list of all the models

            :config: configuration parameters

        Return: 
            a list of optimizers

        """
        raise NotImplementedError

    def data_creator(self, config : dict):
        r"""
        Create DataLoaders.

        Args:
            :config: configuration parameters

        Return:
             (train_loader, val_loader)
        """
        return data_creator(config, training_mode='transition', val_horizon=config['venv_rollout_horizon'], double=True)

    def _setup_componects(self, config : dict):
        r'''setup models, optimizers and dataloaders.'''

        # register data loader for double venv training
        train_loader_train, val_loader_train, train_loader_val, val_loader_val = self.data_creator(config)
        try:
            self._train_loader_train = train.torch.prepare_data_loader(train_loader_train, move_to_device=False)
            self._val_loader_train = train.torch.prepare_data_loader(val_loader_train, move_to_device=False)
            self._train_loader_val = train.torch.prepare_data_loader(train_loader_val, move_to_device=False)
            self._val_loader_val = train.torch.prepare_data_loader(val_loader_val, move_to_device=False)
        except:
            self._train_loader_train = train_loader_train
            self._val_loader_train = val_loader_train
            self._train_loader_val = train_loader_val
            self._val_loader_val = val_loader_val

        self.train_models = self.model_creator(config, self.graph_train)
        for model_index, model in enumerate(self.train_models):
            try:
                self.train_models[model_index] = train.torch.prepare_model(model)
            except:
                self.train_models[model_index] = model.to(self._device)
        self.train_optimizers = self.optimizer_creator(self.train_models, config)

        self.val_models = self.model_creator(config, self.graph_val)
        for model_index, model in enumerate(self.val_models):
            try:
                self.val_models[model_index] = train.torch.prepare_model(model)
            except:
                self.val_models[model_index] = model.to(self._device)
        self.val_optimizers = self.optimizer_creator(self.val_models, config)

    def _register_models_to_graph(self, graph : DesicionGraph, models : List[torch.nn.Module]):
        index = 0
        # register policy nodes 
        for node_name in list(graph.keys()):
            if node_name in graph.transition_map.values(): 
                continue
            node = graph.get_node(node_name)
            if node.node_type == 'network':
                node.set_network(models[index])
                index += 1
        # register transition nodes
        for node_name in graph.transition_map.values():
            node = graph.get_node(node_name)
            if node.node_type == 'network':
                node.set_network(models[index])
                index += 1               
        assert len(models) == index, f'Some models are not registered. Total models: {len(models)}, Registered: {index}.'

    @catch_error
    def __init__(self, config : dict):
        r'''setup everything for training.
        
        Args:
            :config: configuration parameters
        
        '''
        # parse information from config
        self.config = config
        self.train_dataset = ray.get(config['dataset'])
        self.val_dataset = ray.get(config['val_dataset'])
        self._data_buffer = config['venv_data_buffer']
        self._workspace = config["workspace"]
        self._graph = deepcopy(config['graph'])
        self._filename = os.path.join(self._workspace, "train_venv.json")
        self._data_buffer.set_total_trials.remote(config.get("total_num_of_trials", 1))
        self._data_buffer.inc_trial.remote()
        self._least_metric_train = [np.inf] * len(self._graph.metric_nodes)
        self._least_metric_val = [np.inf] * len(self._graph.metric_nodes)
        self.least_val_metric = np.inf
        self.least_train_metric = np.inf
        self._acc = 0
        self._num_venv_list = []

        # get id
        self._ip = ray._private.services.get_node_ip_address()
        
        logger.add(os.path.join(os.path.abspath(self._workspace),"revive.log"))
        
        # set trail seed
        setup_seed(config["global_seed"])

        if 'tag' in self.config:  # create by tune
            tag = self.config['tag']
            logger.info("{} is running".format(tag))
            self._traj_id = int(tag.split('_')[0])
            experiment_dir = os.path.join(self._workspace, 'venv_tune')
            traj_name_list = sorted(os.listdir(experiment_dir),key=lambda x:x[-19:], reverse=True)
            for traj_name in filter(lambda x: "ReviveLog" in x, traj_name_list):
                if len(traj_name.split('_')[1]) == 5: # create by random search or grid search
                    id_index = 3
                else:
                    id_index = 2
                if int(traj_name.split('_')[id_index]) == self._traj_id:
                    self._traj_dir = os.path.join(experiment_dir, traj_name)
                    break
        else:
            self._traj_id = 1
            self._traj_dir = os.path.join(self._workspace, 'venv_train')

        update_env_vars("pattern", "venv")

        # setup constant
        self._stop_flag = False
        self._batch_cnt = 0
        self._epoch_cnt = 0
        self._last_wdist_epoch = 0
        self._wdist_id_train = None
        self._wdist_id_val = None
        self._wdist_ready_train = False
        self._wdist_ready_val = False
        self._wdist_ready = False
        self._use_gpu = self.config["use_gpu"] and torch.cuda.is_available()
        self._device = 'cuda' if self._use_gpu else 'cpu' # fix problem introduced in ray 1.1

        # if "shooting_" in self.config['venv_metric']:
        #     self.config['venv_metric'] = self.config['venv_metric'].replace("shooting_", "rollout_")
        if self.config['venv_metric'] in ["mae","mse","nll"]:
            self.config['venv_metric'] = "rollout_" + self.config['venv_metric']
        
        # prepare for training
        self.graph_train = deepcopy(self._graph)
        self.graph_val = deepcopy(self._graph)
        self._setup_componects(config)

        # register models to graph
        self._register_models_to_graph(self.graph_train, self.nodes_models_train)
        self._register_models_to_graph(self.graph_val, self.nodes_models_val)

        self.total_dim = 0
        for node_name in self._graph.metric_nodes:
            self.total_dim += self.config['total_dims'][node_name]['input']

        self.nodes_dim_name_map = {}
        for node_name in list(self._graph.nodes) + list(self._graph.leaf):
            node_dims = []
            for node_dim in self._graph.descriptions[node_name]:
                node_dims.append(list(node_dim.keys())[0])
            self.nodes_dim_name_map[node_name] = node_dims
        logger.info(f"Nodes : {self.nodes_dim_name_map}")

        self.graph_to_save_train = self.graph_train
        self.graph_to_save_val = self.graph_val
        self.best_graph_train = deepcopy(self.graph_train)
        self.best_graph_val = deepcopy(self.graph_val)

        if self._traj_dir.endswith("venv_train"):
            if "venv.pkl" in os.listdir(self._traj_dir):
                logger.info("Find existing checkpoint, Back up existing model.")
                try:
                    self._traj_dir_bak = self._traj_dir+"_bak"
                    self._filename_bak = self._filename+"_bak"
                    shutil.copytree(self._traj_dir, self._traj_dir_bak )
                    shutil.copy(self._filename, self._filename_bak)
                except:
                    pass

        self._save_models(self._traj_dir)
        self._update_metric()

        self.global_step = 0

    def nan_in_grad(self):
        if hasattr(self, "nums_nan_in_grad"):
            self.nums_nan_in_grad += 1
        else:
            self.nums_nan_in_grad = 1
        if self.nums_nan_in_grad > 100:
            self._stop_flag = True
            logger.warning(f'Find too many nan in loss. Early stop.')

    def _early_stop(self, info : dict):
        info["stop_flag"] = self._stop_flag
        return info

    def _update_metric(self):
        ''' update metric to data buffer '''
        # NOTE: Very large model (e.g. 1024 x 4 LSTM) cannot be updated.
        try:
            venv_train = torch.load(os.path.join(self._traj_dir, 'venv_train.pt'), map_location='cpu')
            venv_val = torch.load(os.path.join(self._traj_dir, 'venv_val.pt'), map_location='cpu')
        except:
            venv_train = venv_val = None

        try:
            metric = self.least_metric
        except:
            metric = np.sum(self._least_metric_train) / self.total_dim
        if 'mse' in self.metric_name or 'mae' in self.metric_name:
            acc = (self.config['max_distance'] - np.log(metric)) / (self.config['max_distance'] - self.config['min_distance'])
        else:
            acc = (self.config['max_distance'] - metric) / (self.config['max_distance'] - self.config['min_distance'])
        
        acc = min(max(acc,0),1)
        acc = 0.5 + (0.4*acc)
        if acc == 0.5:
            acc = 0      
        self._acc = acc
        self._data_buffer.update_metric.remote(self._traj_id, {
            "metric": metric,
            "acc" : acc,
            "ip": self._ip,
            "venv_train" : venv_train,
            "venv_val" : venv_val,
            "traj_dir" : self._traj_dir,
        })
        self._data_buffer.write.remote(self._filename)
        if metric == ray.get(self._data_buffer.get_least_metric.remote()):
            self._save_models(self._workspace, with_env=False)

    def _save_models(self, path: str, with_env:bool=True, model_prefixes:str=""):
        """ 
            param: path, where to save the models
            param: with_env, whether to save venv along with the models
        """
        if model_prefixes:
            model_prefixes = model_prefixes + "_"
        best_graph_train = deepcopy(self.graph_train)
        best_graph_val   = deepcopy(self.graph_val)

        best_graph_train.reset()
        for node_name in best_graph_train.keys():
            node = best_graph_train.get_node(node_name)
            if node.node_type == 'network':
                network = deepcopy(node.get_network()).cpu()
                torch.save(network, os.path.join(path, node_name + '_train.pt'))

        best_graph_val.reset()
        for node_name in best_graph_val.keys():
            node = best_graph_val.get_node(node_name)
            if node.node_type == 'network':
                network = deepcopy(node.get_network()).cpu()
                torch.save(network, os.path.join(path, node_name + '_val.pt'))

        if with_env:
            best_graph_train = deepcopy(best_graph_train).to("cpu")
            venv_train = VirtualEnvDev(best_graph_train)
            torch.save(venv_train, os.path.join(path, "venv_train.pt"))

            best_graph_val = deepcopy(best_graph_val).to("cpu")
            venv_val = VirtualEnvDev(best_graph_val)
            torch.save(venv_val, os.path.join(path, "venv_val.pt"))

            venv = VirtualEnv([venv_train, venv_val])
            with open(os.path.join(path, model_prefixes + 'venv.pkl'), 'wb') as f:
                pickle.dump(venv, f)

            venv_list = ray.get(self._data_buffer.get_best_venv.remote()) #self._data_buffer.get_best_venv() #
            with open(os.path.join(path, model_prefixes +'ensemble_env.pkl'), 'wb') as f:
                pickle.dump(venv_list, f)

    def _load_best_models(self):
        best_graph_train = deepcopy(self.graph_train)
        best_graph_val   = deepcopy(self.graph_val)
        for node_name in best_graph_train.keys():
            best_node = best_graph_train.get_node(node_name)
            current_node = self.graph_train.get_node(node_name)
            if best_node.node_type == 'network':
                current_node.get_network().load_state_dict(best_node.get_network().state_dict())

        for node_name in best_graph_val.keys():
            best_node = best_graph_val.get_node(node_name)
            current_node = self.graph_val.get_node(node_name)
            if best_node.node_type == 'network':
                current_node.get_network().load_state_dict(best_node.get_network().state_dict())

    @torch.no_grad()
    def _log_histogram(self, expert_data, generated_data, scope='valEnv_on_trainData'):
        if scope == 'valEnv_on_trainData':
            graph = self.graph_val
        else:
            graph = self.graph_train  

        info = {}
        for node_name in graph.keys():
            # compute values 
            if graph.get_node(node_name).node_type == 'network':
                node_dist = graph.compute_node(node_name, expert_data)
                expert_action = expert_data[node_name]
                generated_action = node_dist.sample()
                policy_std = node_dist.std
            else:
                expert_action = expert_data[node_name]
                generated_action = graph.compute_node(node_name, expert_data)
                policy_std = torch.zeros_like(generated_action)         

            # make logs
            error = expert_action - generated_action
            expert_action = expert_action.cpu()
            generated_action = generated_action.cpu()
            error = error.cpu()
            policy_std = policy_std.cpu()
            for i in range(error.shape[-1]):
                info[f'{node_name}_dim{i}_{scope}/error'] = error.select(-1, i)
                info[f'{node_name}_dim{i}_{scope}/expert'] = expert_action.select(-1, i)
                info[f'{node_name}_dim{i}_{scope}/sampled'] = generated_action.select(-1, i)
                info[f'{node_name}_dim{i}_{scope}/sampled_std'] = policy_std.select(-1, i)

        return info

    def _env_test(self, scope : str = 'train'):
        # pick target policy
        graph = self.graph_train if scope == 'train' else self.graph_val
        node = graph.get_node(self.config['target_policy_name'])

        env = create_env(self.config['task'])
        if env is None:
            return {}

        graph.reset()
        node = deepcopy(node)
        node = node.to('cpu')
        policy = PolicyModelDev(node)
        policy = PolicyModel(policy)

        reward, length = test_on_real_env(env, policy)

        return {
            f"{self.NAME}/real_reward_{scope}" : reward,
            f"{self.NAME}/real_length_{scope}" : length,
        }

    def _mse_test(self, expert_data, generated_data, scope='valEnv_on_trainData'):
        info = {}

        if not self.config['mse_test'] and not 'mse' in self.metric_name:
            return info

        if 'mse' in self.metric_name:
            self.graph_to_save_train = self.graph_train
            self.graph_to_save_val = self.graph_val

        if scope == 'valEnv_on_trainData':
            graph = self.graph_val
        else:
            graph = self.graph_train    

        graph.reset()        

        new_data = Batch({name : expert_data[name] for name in graph.leaf})
        total_mse = 0
        for node_name in graph.keys():
            if node_name + "_isnan_index_" in expert_data.keys():
                isnan_index = 1 - torch.mean(expert_data[node_name + "_isnan_index_"]) 
            else:
                isnan_index = None
            if graph.get_node(node_name).node_type == 'network':
                node_dist = graph.compute_node(node_name, new_data)
                new_data[node_name] = node_dist.mode
            else:
                new_data[node_name] = graph.compute_node(node_name, new_data)
                continue
            if node_name in graph.metric_nodes:
                if isnan_index is not None:
                    node_mse = (((new_data[node_name] - expert_data[node_name])*isnan_index) ** 2).sum(dim=-1).mean()
                else:
                    node_mse = ((new_data[node_name] - expert_data[node_name]) ** 2).sum(dim=-1).mean()
                total_mse += node_mse.item()
                info[f"{self.NAME}/{node_name}_one_step_mse_{scope}"] = node_mse.item()
        info[f"{self.NAME}/average_one_step_mse_{scope}"] = total_mse / self.total_dim

        mse_error = 0
        for node_name in graph.metric_nodes:
            if node_name + "_isnan_index_" in expert_data.keys():
                isnan_index = 1 - torch.mean(expert_data[node_name + "_isnan_index_"]) 
            else:
                isnan_index = None
            if isnan_index is not None:
                policy_rollout_mse = (((expert_data[node_name] - generated_data[node_name])*isnan_index) ** 2).sum(dim=-1).mean()
            else:
                policy_rollout_mse = ((expert_data[node_name] - generated_data[node_name]) ** 2).sum(dim=-1).mean()
            mse_error += policy_rollout_mse.item()
            info[f"{self.NAME}/{node_name}_rollout_mse_{scope}"] = policy_rollout_mse.item()
        info[f"{self.NAME}/average_rollout_mse_{scope}"] = mse_error / self.total_dim

        return info

    def _nll_test(self, expert_data, generated_data, scope='valEnv_on_trainData'): # negative log likelihood
        info = {}

        if not self.config['nll_test'] and not 'nll' in self.metric_name:
            return info

        if 'nll' in self.metric_name:
            self.graph_to_save_train = self.graph_train
            self.graph_to_save_val = self.graph_val

        if scope == 'valEnv_on_trainData':
            graph = self.graph_val
        else:
            graph = self.graph_train 

        graph.reset()

        new_data = Batch({name : expert_data[name] for name in graph.leaf})
        total_nll = 0
        for node_name in graph.keys():
            if node_name + "_isnan_index_" in expert_data.keys():
                isnan_index = 1 - torch.mean(expert_data[node_name + "_isnan_index_"],axis=-1) 
            else:
                isnan_index = None

            if node_name in graph.learnable_node_names and node_name in graph.metric_nodes:
                node_dist = graph.compute_node(node_name, new_data)
                new_data[node_name] = node_dist.mode
            else:
                new_data[node_name] = expert_data[node_name]
                continue

            if isnan_index is not None:
                node_nll = - (node_dist.log_prob(expert_data[node_name])*isnan_index).mean()
            else:
                node_nll = - node_dist.log_prob(expert_data[node_name]).mean()
            total_nll += node_nll.item()
            info[f"{self.NAME}/{node_name}_one_step_nll_{scope}"] = node_nll.item()

        
        info[f"{self.NAME}/average_one_step_nll_{scope}"] = total_nll / self.total_dim

        total_nll = 0
        for node_name in graph.metric_nodes:
            if node_name + "_isnan_index_" in expert_data.keys():
                isnan_index = 1 - expert_data[node_name + "_isnan_index_"] 
            else:
                isnan_index = None
            if node_name in graph.learnable_node_names:
                node_dist = graph.compute_node(node_name, expert_data)
                if isnan_index is not None:
                    policy_nll = - (node_dist.log_prob(expert_data[node_name]*isnan_index)).mean()
                else:
                    policy_nll = - node_dist.log_prob(expert_data[node_name]).mean()
                total_nll += policy_nll.item()
                info[f"{self.NAME}/{node_name}_rollout_nll_{scope}"] = policy_nll.item()
            else:
                total_nll += 0
                info[f"{self.NAME}/{node_name}_rollout_nll_{scope}"] = 0

        info[f"{self.NAME}/average_rollout_nll_{scope}"] = total_nll / self.total_dim

        return info
        
    def _mae_test(self, expert_data, generated_data, scope='valEnv_on_trainData'):
        info = {}

        if not self.config['mae_test'] and not 'mae' in self.metric_name:
            return info

        if 'mae' in self.metric_name:
            self.graph_to_save_train = self.graph_train
            self.graph_to_save_val = self.graph_val

        if scope == 'valEnv_on_trainData':
            graph = self.graph_val
        else:
            graph = self.graph_train 

        graph.reset()

        new_data = Batch({name : expert_data[name] for name in graph.leaf})
        total_mae = 0
        for node_name in graph.keys():
            if node_name + "_isnan_index_" in expert_data.keys():
                isnan_index = 1 - torch.mean(expert_data[node_name + "_isnan_index_"]) 
            else:
                isnan_index = None

            if graph.get_node(node_name).node_type == 'network':
                node_dist = graph.compute_node(node_name, new_data)
                new_data[node_name] = node_dist.mode
            else:
                new_data[node_name] = graph.compute_node(node_name, new_data)
                continue
            if node_name in graph.metric_nodes:
                if isnan_index is not None:
                    node_mae = ((new_data[node_name] - expert_data[node_name])*isnan_index).abs().sum(dim=-1).mean()
                else:
                    node_mae = (new_data[node_name] - expert_data[node_name]).abs().sum(dim=-1).mean()
                total_mae += node_mae.item()
                info[f"{self.NAME}/{node_name}_one_step_mae_{scope}"] = node_mae.item()

        info[f"{self.NAME}/average_one_step_mae_{scope}"] = total_mae / self.total_dim

        mae_error = 0
        for node_name in graph.keys():
            if node_name in graph.metric_nodes:
                if node_name + "_isnan_index_" in expert_data.keys():
                    isnan_index = 1 - torch.mean(expert_data[node_name + "_isnan_index_"]) 
                else:
                    isnan_index = None
                if isnan_index is not None:
                    policy_shooting_error = (torch.abs(expert_data[node_name] - generated_data[node_name])*isnan_index).sum(dim=-1).mean()
                else:
                    policy_shooting_error = torch.abs(expert_data[node_name] - generated_data[node_name]).sum(dim=-1).mean()
                mae_error += policy_shooting_error.item()
                info[f"{self.NAME}/{node_name}_rollout_mae_{scope}"] = policy_shooting_error.item()

                # TODO: plot rollout error
                # rollout_error = torch.abs(expert_data[node_name] - generated_data[node_name]).reshape(expert_data.shape[0],-1).mean(dim=-1)

        info[f"{self.NAME}/average_rollout_mae_{scope}"] = mae_error / self.total_dim

        return info

    def _wdist_test(self, expert_data, generated_data, scope='valEnv_on_trainData'):
        info = {}

        if not self.config['wdist_test'] and not 'wdist' in self.metric_name:
            return info

        if 'wdist' in self.metric_name:
            self.graph_to_save_train = self.graph_train
            self.graph_to_save_val = self.graph_val

        if scope == 'valEnv_on_trainData':
            graph = self.graph_val
        else:
            graph = self.graph_train 

        graph.reset()

        wdist_error = []
        for node_name in graph.keys():
            if node_name in graph.metric_nodes:
                # TODO: support isnan_index
                node_dim = expert_data[node_name].shape[-1]
                if node_name + "_isnan_index_" in expert_data.keys():
                    isnan_index = 1 - torch.mean(expert_data[node_name + "_isnan_index_"],axis=-1) 
                else:
                    isnan_index = None
                wdist = [stats.wasserstein_distance(expert_data[node_name].reshape(-1, expert_data[node_name].shape[-1])[..., dim].cpu().numpy(),
                                                    generated_data[node_name].reshape(-1, generated_data[node_name].shape[-1])[..., dim].cpu().numpy())
                                for dim in range(node_dim)]
                wdist = np.sum(wdist)
                info[f"{self.NAME}/{node_name}_wdist_{scope}"] = wdist
                
                wdist_error.append(wdist)

        info[f"{self.NAME}/average_wdist_{scope}"] = np.sum(wdist_error) / self.total_dim
        return info
    
    @catch_error
    def before_train_epoch(self):
        update_env_vars("venv_epoch",self._epoch_cnt)
        
    @catch_error
    def train_epoch(self):
        info = dict()
 
        r"""Define the training process for an epoch."""
        self._epoch_cnt += 1
        logger.info(f"Train epoch : {self._epoch_cnt} ")

        # switch to training mode
        if hasattr(self, "model"):
            self.model.train()
        if hasattr(self, "models"):
            for _model in self.models:
                _model.train()

        metric_meters_train = AverageMeterCollection()

        for batch_idx, batch in enumerate(iter(self._train_loader_train)):
            batch_info = {
                "batch_idx": batch_idx,
                "global_step": self.global_step
            }
            batch_info.update(info)
            metrics = self.train_batch(batch, batch_info=batch_info, scope='train')

            metric_meters_train.update(metrics, n=metrics.pop(NUM_SAMPLES, 1))
            self.global_step += 1

        metric_meters_val = AverageMeterCollection()

        for batch_idx, batch in enumerate(iter(self._val_loader_train)):
            batch_info = {
                "batch_idx": batch_idx,
                "global_step": self.global_step
            }
            batch_info.update(info)
            metrics = self.train_batch(batch, batch_info=batch_info, scope='val')

            metric_meters_val.update(metrics, n=metrics.pop(NUM_SAMPLES, 1))
            self.global_step += 1        

        info = metric_meters_train.summary()
        info.update(metric_meters_val.summary())
        return {k : info[k] for k in filter(lambda k: not k.startswith('last'), info.keys())}

    @catch_error
    def validate(self):
        logger.info(f"Epoch : {self._epoch_cnt} ")
        info = dict()
        
        r"""Define the validate process after train one epoch."""
        if hasattr(self, "model"):
            self.model.eval()
        if hasattr(self, "models"):
            for _model in self.models:
                _model.eval()

        self.logged_histogram = {
            'valEnv_on_trainData' : False,
            'trainEnv_on_valData' : False,
        }

        with torch.no_grad():
            metric_meters_train = AverageMeterCollection()
            for batch_idx, batch in enumerate(iter(self._train_loader_val)):
                batch_info = {"batch_idx": batch_idx}
                batch_info.update(info)
                metrics = self.validate_batch(batch, batch_info, 'valEnv_on_trainData')
                metric_meters_train.update(metrics, n=metrics.pop(NUM_SAMPLES, 1))
                if batch_idx > 128:
                    break
            metric_meters_val = AverageMeterCollection()
            for batch_idx, batch in enumerate(iter(self._val_loader_val)):
                batch_info = {"batch_idx": batch_idx}
                batch_info.update(info)
                metrics = self.validate_batch(batch, batch_info, 'trainEnv_on_valData')
                metric_meters_val.update(metrics, n=metrics.pop(NUM_SAMPLES, 1))
                if batch_idx > 128:
                    break

        info = metric_meters_train.summary()
        info.update(metric_meters_val.summary())
        info = {k : info[k] for k in filter(lambda k: not k.startswith('last'), info.keys())}

        # run test on real environment
        if (not self.config['real_env_test_frequency'] == 0) and \
            self._epoch_cnt % self.config['real_env_test_frequency'] == 0:
            info.update(self._env_test('train'))
            info.update(self._env_test('val'))

        if self._epoch_cnt >= self.config['save_start_epoch']:
            need_update = []
            for i, node_name in enumerate(self.graph_train.metric_nodes):
                i = self.graph_train.metric_nodes.index(node_name)
                if info[f"{self.NAME}/{node_name}_{self.config['venv_metric']}_trainEnv_on_valData"] < self._least_metric_train[i]:
                    self._least_metric_train[i] = info[f"{self.NAME}/{node_name}_{self.config['venv_metric']}_trainEnv_on_valData"]
                    # [ SIGNIFICANT OTHER ] deepcopy is necessary, otherwise, the best is always the same to the current!
                    self.best_graph_train.nodes[node_name] = deepcopy(self.graph_to_save_train.get_node(node_name))
                    need_update.append(True)
                else:
                    need_update.append(False)

            for i, node_name in enumerate(self.graph_val.metric_nodes):
                i = self.graph_val.metric_nodes.index(node_name)
                if info[f"{self.NAME}/{node_name}_{self.config['venv_metric']}_valEnv_on_trainData"] < self._least_metric_val[i]:
                    self._least_metric_val[i] = info[f"{self.NAME}/{node_name}_{self.config['venv_metric']}_valEnv_on_trainData"]
                    # [ SIGNIFICANT OTHER ] deepcopy is necessary, otherwise, the best is always the same to the current!
                    self.best_graph_val.nodes[node_name] = deepcopy(self.graph_to_save_val.get_node(node_name))
                    need_update.append(True)
                else:
                    need_update.append(False)

            if self.config["save_by_node"]:
                info["least_metric"] = np.sum(self._least_metric_train) / self.total_dim
                self.least_metric = info["least_metric"]
                if True in need_update:
                    self._save_models(self._traj_dir)
                    self._update_metric()

        info["stop_flag"] = self._stop_flag
        info['now_metric'] = info[f'{self.metric_name}_trainEnv_on_valData']
        
        if not self.config["save_by_node"]:
            now_val_metric = info[f'{self.metric_name}_valEnv_on_trainData']
            if self._epoch_cnt >= self.config['save_start_epoch']:
                #if (info["now_metric"] <= self.least_train_metric or now_val_metric <= self.least_val_metric):
                if info["now_metric"] <= self.least_train_metric:
                    self._save_models(self._traj_dir)
                    self._update_metric()
                    if info["now_metric"] <= self.least_train_metric:
                        self.least_train_metric = info["now_metric"]
                    if now_val_metric <= self.least_val_metric:
                        self.least_val_metric = now_val_metric

            info["least_metric"] = self.least_train_metric
            self.least_metric = self.least_train_metric
            
            if self.config["venv_save_frequency"]:
                if self._epoch_cnt % self.config["venv_save_frequency"] == 0:
                    self._save_models(self._traj_dir, model_prefixes = str(self._epoch_cnt)) 
                
            
            # Save the k env for every task
            _k = 1
            while self._epoch_cnt % _k == 0:
                if len(self._num_venv_list) < self.config['num_venv_store']:
                    self._num_venv_list.append(info["now_metric"])
                    pass
                elif info["now_metric"] < np.max(self._num_venv_list):
                    _del_index = np.argmax(self._num_venv_list)
                    self._data_buffer.delet_deque_item.remote(self._traj_id, _del_index)
                    self._num_venv_list.pop(_del_index)
                    self._num_venv_list.append(info["now_metric"])
                    pass
                else:
                    break

                _best_graph_train = deepcopy(self.graph_train).to("cpu")
                _best_graph_val   = deepcopy(self.graph_val).to("cpu")
                _best_graph_train.reset()
                _best_graph_val.reset()
                _venv_train = VirtualEnvDev(_best_graph_train)
                _venv_val = VirtualEnvDev(_best_graph_val)
                self._data_buffer.update_venv_deque_dict.remote(self._traj_id, _venv_train, _venv_val)
                break

        for k in list(info.keys()):
            if self.NAME in k:
                v = info.pop(k)
                info['VAL_' + k] = v

        '''plot histogram when training is finished'''
        # [ OTHER ] more frequent valuation
        if self._stop_flag or self._epoch_cnt % self.config["rollout_plt_frequency"] == 0:
            if self.config["rollout_plt_frequency"] > 0:
                histogram_path = os.path.join(self._traj_dir, 'histogram')
                if not os.path.exists(histogram_path):
                    os.makedirs(histogram_path)
                try:
                    save_histogram(histogram_path, self.best_graph_train, self._train_loader_val, device=self._device, scope='train')
                    save_histogram(histogram_path, self.best_graph_val, self._val_loader_val, device=self._device, scope='val')

                    if self.config["rollout_dataset_mode"] == "validate":
                        rollout_dataset = self.val_dataset
                    else:
                        rollout_dataset = self.train_dataset
                    # save rolllout action image
                    rollout_save_path = os.path.join(self._traj_dir, 'rollout_images')
                    nodes_map = deepcopy(self.nodes_dim_name_map)
                    # del step_node
                    if "step_node_" in nodes_map.keys():
                        nodes_map.pop("step_node_")
                    save_rollout_action(rollout_save_path, self.best_graph_train, self._device, rollout_dataset, deepcopy(nodes_map))
                    # [ OTHER ] not only plotting the best model, but also plotting the result of the current model
                    rollout_save_path = os.path.join(self._traj_dir, 'rollout_images_current')
                    save_rollout_action(rollout_save_path, self.graph_train, self._device, rollout_dataset, deepcopy(nodes_map))
                except Exception as e:
                    logger.warning(e)
            else:
                logger.info("Don't plot images.")

        if self._epoch_cnt == 1:
            try:
                info = self._load_checkpoint(info)
            except:
                logger.info("Don't Load checkpoint!")
            if hasattr(self, "_traj_dir_bak") and os.path.exists(self._traj_dir_bak):
                shutil.rmtree(self._traj_dir_bak)
                os.remove(self._filename_bak)

        if self._stop_flag:
            if self.config["plt_response_curve"]:
                response_curve_path = os.path.join(self._traj_dir, 'response_curve')
                if not os.path.exists(response_curve_path):
                    os.makedirs(response_curve_path)
                dataset = ray.get(self.config['dataset'])
                plot_response_curve(response_curve_path, self.best_graph_train, self.best_graph_val, dataset=dataset.data, device=self._device)
            try:
                customer_uploadTrainLog(self.config["trainId"],
                                        os.path.join(os.path.abspath(self._workspace),"revive.log"),
                                        "train.simulator",
                                        "success",
                                        self._acc,
                                        self.config["accessToken"])
            except Exception as e:
                error_message = traceback.format_exc()
                error_message = ""
                logger.info('Detect error:{}, Error Message: {}'.format(e,error_message))

        return info

    def _load_checkpoint(self,info):
        if self._traj_dir.endswith("venv_train"):
            self._load_models(self._traj_dir_bak)
            with open(self._filename_bak, 'r') as f:
                train_log = json.load(f)
            metric = train_log["metrics"]["1"]
            venv_train = torch.load(os.path.join(self._traj_dir_bak, 'venv_train.pt'), map_location='cpu')
            venv_val = torch.load(os.path.join(self._traj_dir_bak, 'venv_val.pt'), map_location='cpu')
            self._data_buffer.update_metric.remote(self._traj_id, {
                "metric": metric["metric"],
                "acc" : metric["acc"],
                "ip": self._ip,
                "venv_train" : venv_train,
                "venv_val" : venv_val,
                "traj_dir" : self._traj_dir,
            })

            self._data_buffer.write.remote(self._filename)
            self._save_models(self._workspace, with_env=False)
            return info
        else:
            with open(os.path.join(self._traj_dir,"params.json"), 'r') as f:
                params = json.load(f)

            experiment_dir = os.path.dirname(self._traj_dir)
            dir_name_list = [dir_name for dir_name in os.listdir(experiment_dir) if dir_name.startswith("ReviveLog")]
            dir_name_list = sorted(dir_name_list,key=lambda x:x[-19:], reverse=False)

            for dir_name in dir_name_list:
                dir_path = os.path.join(experiment_dir, dir_name)
                if dir_path == self._traj_dir:
                    break
                if os.path.isdir(dir_path):
                    params_json_path = os.path.join(dir_path, "params.json")
                    if os.path.exists(params_json_path):
                        with open(params_json_path, 'r') as f:
                            history_params = json.load(f)
                        if history_params == params:
                            result_json_path = os.path.join(dir_path, "result.json")

                            with open(result_json_path, 'r') as f:
                                history_result = []
                                for line in f.readlines():
                                    history_result.append(json.loads(line))
                            if history_result:# and history_result[-1]["stop_flag"]:
                                for k in info.keys():
                                    if k in history_result[-1].keys():
                                        info[k] = history_result[-1][k]

                                self.least_metric = info["least_metric"]
                                # load model
                                logger.info("Find exist checkpoint, Load the model.")
                                self._load_models(dir_path)
                                self._save_models(self._traj_dir)
                                self._update_metric()
                                # check early stop
                                self._stop_flag = history_result[-1]["stop_flag"]
                                info["stop_flag"] = self._stop_flag
                                logger.info("Load checkpoint success!")
                                break
            return info

    def _load_models(self, path : str, with_env : bool = True):
        """ 
            param: path, where to load the models
            param: with_env, whether to load venv along with the models
        """
        for node_name in self.best_graph_train.keys():
            best_node = self.best_graph_train.get_node(node_name)
            if best_node.node_type == 'network':
                best_node.get_network().load_state_dict(torch.load(os.path.join(path, node_name + '_train.pt')).state_dict())

        for node_name in self.best_graph_val.keys():
            best_node = self.best_graph_val.get_node(node_name)
            if best_node.node_type == 'network':
                best_node.get_network().load_state_dict(torch.load(os.path.join(path, node_name + '_train.pt')).state_dict())


    def train_batch(self, expert_data, batch_info, scope='train'):
        r"""Define the training process for an batch data."""
        raise NotImplementedError

    def validate_batch(self, expert_data, batch_info, scope='valEnv_on_trainData'):
        r"""Define the validate process for an batch data.

        Args:
            expert_data: The batch offline Data.

            batch_info: A batch info dict.

            scope: if ``scope=valEnv_on_trainData`` means training data test on the model trained by validation dataset.
        """
        info = {}

        if scope == 'valEnv_on_trainData':
            graph = self.graph_val
        else:
            graph = self.graph_train

        expert_data.to_torch(device=self._device)

        traj_length = expert_data.shape[0]

        # data is generated by taking the most likely action
        sample_fn = lambda dist: dist.mode
        generated_data = generate_rollout(expert_data, graph, traj_length, sample_fn, clip=True)

        for node_name in graph.nodata_node_names:
            assert node_name not in expert_data.keys() 
            assert node_name in generated_data.keys()
            expert_data[node_name] = generated_data[node_name] 
        info.update(self._nll_test(expert_data, generated_data, scope))
        info.update(self._mse_test(expert_data, generated_data, scope))
        info.update(self._mae_test(expert_data, generated_data, scope))
        info.update(self._wdist_test(expert_data, generated_data, scope))

        # log histogram info
        if (not self.config['histogram_log_frequency'] == 0) and \
            self._epoch_cnt % self.config['histogram_log_frequency'] == 0 and \
            not self.logged_histogram[scope]:
            info.update(self._log_histogram(expert_data, generated_data, scope))   
            self.logged_histogram[scope] = True

        return info


class VenvAlgorithm:
    ''' Class use to manage venv algorithms '''
    def __init__(self, algo : str, workspace: str =None):
        self.algo = algo
        self.workspace = workspace
        if self.algo == "revive" or self.algo == "revive_p" or self.algo == "revivep" or self.algo == "revive_ppo":
            self.algo = "revive_p"
        elif self.algo == "revive_t" or self.algo == "revivet"  or self.algo == "revive_td3":
            self.algo = "revivet"
        elif self.algo == "bc":
            self.algo = "bc"
        else:
            raise NotImplementedError

        try:
            self.algo_module = importlib.import_module(f'revive.dist.algo.venv.{self.algo}')
            logger.info(f"Import encryption venv algorithm module -> {self.algo}!")
        except:
            self.algo_module = importlib.import_module(f'revive.algo.venv.{self.algo}')
            logger.info(f"Import venv algorithm module -> {self.algo}!")

        # Assume there is only one operator other than VenvOperator
        for k in self.algo_module.__dir__():
            if 'Operator' in k and not k == 'VenvOperator':
                self.operator = getattr(self.algo_module, k)

        self.operator_config = {}

    def get_train_func(self, config):
        from revive.utils.tune_utils import VALID_SUMMARY_TYPES
        from torch.utils.tensorboard import SummaryWriter
        from ray.air import session

        def train_func(config):
            for k,v in self.operator_config.items():
                if not k in config.keys():
                    config[k] = v
            algo_operator = self.operator(config)
            writer = SummaryWriter(algo_operator._traj_dir)
            epoch = 0
            while True:
                algo_operator.before_train_epoch()
                train_stats = algo_operator.train_epoch()
                val_stats = algo_operator.validate()
                session.report({"mean_accuracy": algo_operator._acc,
                                "least_metric": val_stats["least_metric"],
                                "now_metric": val_stats["now_metric"],
                                "stop_flag": val_stats["stop_flag"]})

                # write tensorboard
                for k, v in [*train_stats.items(), *val_stats.items()]:
                    if type(v) in VALID_SUMMARY_TYPES:
                        writer.add_scalar(k, v, global_step=epoch)
                    elif isinstance(v, torch.Tensor):
                        v = v.view(-1)
                        writer.add_histogram(k, v, global_step=epoch)
                writer.flush()
                # check stop_flag
                train_stats.update(val_stats)
                if train_stats.get('stop_flag', False):
                    break
                epoch += 1

        return train_func 

    def get_trainer(self, config):
        try:
            train_func = self.get_train_func(config)
            from ray.air.config import ScalingConfig
            trainer = TorchTrainer(
                train_func,
                train_loop_config=config,
                scaling_config=ScalingConfig(num_workers=config['workers_per_trial'], use_gpu=config['use_gpu']),
            )
            return trainer 
        except Exception as e:
            logger.error('Detect Error: {}'.format(e))
            raise e
    
    def get_trainable(self, config):
        try:
            trainer = self.get_trainer(config)
            trainable = trainer.as_trainable()

            return trainable 
        except Exception as e:
            logger.error('Detect Error: {}'.format(e))
            raise e

    def get_parameters(self, command=None):
        try:
            return self.operator.get_parameters(command)
        except AttributeError:
            raise AttributeError("Custom algorithm need to implement `get_parameters`")
        except Exception as e:
            logger.error('Detect Unknown Error:'.format(e))
            raise e

    def get_tune_parameters(self, config):
        try:
            self.operator_config = config
            return self.operator.get_tune_parameters(config)
        except AttributeError:
            raise AttributeError("Custom algorithm need to implement `get_tune_parameters`")
        except Exception as e:
            logger.error('Detect Unknown Error:'.format(e))
            raise e