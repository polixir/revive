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
import yaml
import numpy as np
from copy import deepcopy
from revive.utils.common_utils import load_data
from revive.utils.causal_discovery_utils import ClassicalDiscovery


class CausalGraph:
    def __init__(self, data_file, yaml_file, seed=1024):
        """Finding causal graphs using causal discovery algorithms

        Args:
            data_file : *.npz file path or *.h5 file path

            yaml_fle : *.yaml file path

            seed : random seed

        """
        with open(yaml_file, 'r', encoding='UTF-8') as f:
            self.raw_config = yaml.load(f, Loader=yaml.FullLoader)

        data = load_data(data_file)
        self.seed(seed)

        assert "obs" in data.keys()
        assert "action" in data.keys()

        if "next_obs" not in data.keys():
            end_indexes = data['index'].astype(int)
            start_indexes = np.concatenate([np.array([0]), end_indexes[:-1]])
            curr_obs = []
            next_obs = []
            action = []
            for start, end in zip(start_indexes, end_indexes):
                curr_obs.append(data["obs"][start:end-1])
                next_obs.append(data["obs"][start+1:end])
                action.append(data["action"][start:end-1])

            self.obs = np.concatenate(curr_obs, axis=0)
            self.next_obs = np.concatenate(next_obs, axis=0)
            self.action = np.concatenate(action, axis=0)
        else:
            self.obs = data["obs"]
            self.next_obs = data["next_obs"]
            self.action = data["action"]

        self.data = data
        self.obs_dims = self.obs.shape[1]
        self.action_dims = self.action.shape[1]

        self.algo_cls = ClassicalDiscovery

    def seed(self, seed):
        np.random.seed(seed)

    def fit(self, sample_size=-1):
        """Fit using causal discovery algorithms

        Args:

            sample_size : Limit the number of samples used.
                          The more the number of samples,
                          the longer the training time it takes,
                          -1 means use all samples.

        """

        if sample_size == -1:
            self.algo = self.algo_cls()
        elif sample_size >= 1:
            self.algo = self.algo_cls(limit=int(sample_size))
        else:
            raise ValueError(f"The sample_size should be an integer greater \
                             than or equal to -1. \
                             It should not be {sample_size}")

        data = {"obs": self.obs,
                "action": self.action,
                "next_obs": self.next_obs}

        self.algo.fit(deepcopy(data), fit_transition=True)

    @property
    def causal_graph(self):
        return self.algo.graph.graph

    @property
    def causal_graph_threshold(self):
        return self.algo.graph.thresh_info

    @property
    def causal_matrix(self):
        return self.algo.graph.get_adj_matrix()

    def threshold_transform(self, threshold):
        if threshold == 0.5:
            threshold = self.causal_graph_threshold["common"]
        elif threshold < 0.5:
            min_threshold = self.causal_graph_threshold["min"]
            common_threshold = self.causal_graph_threshold["common"]

            threshold = (common_threshold - min_threshold) * threshold

        else:
            max_threshold = self.causal_graph_threshold["max"]
            common_threshold = self.causal_graph_threshold["common"]

            threshold = (max_threshold - common_threshold) * (threshold - 0.5) + common_threshold

        threshold = min(max(self.causal_graph_threshold["min"], threshold), self.causal_graph_threshold["max"])

        return threshold

    def causal_binary_matrix(self, threshold=None):
        """Convert the causality matrix to a
        two-dimensional connectivity diagram

        Args:

            threshold : Causal truncation threshold,
                        only greater than or equal to
                        this value is considered to have
                        a causal relationship,
                        using 1 means there is a
                        causal relationship,
                        0 means there is no
                        causal relationship.

        Return:

            causal_binary_matrix:  [[0, 0, 1, 1],
                                    [0, 0, 1, 1],
                                    [0, 0, 1, 1],
                                    [0, 0, 1, 1]]

        """
        if threshold is None:
            threshold = 0.5

        threshold = self.threshold_transform(threshold)
        return self.algo.graph.get_binary_adj_matrix(threshold)

    def decision_graph(self, npz_file, yaml_file, threshold=None):
        """Generate decision flow graph for use by REVIVE SDK

        Args:
            yaml_file : The address where the newly
                        generated yaml file is saved.

            npz_file : The address where the newly 
                       generated npz file is saved.

            threshold : Causal truncation threshold,
                        only greater than or equal to
                        this value is considered to have
                        a causal relationship,
                        using 1 means there is a causal
                        relationship, 0 means there is no
                        causal relationship.

        """

        causal_binary_matrix = self.causal_binary_matrix(threshold)
        obs_nodes = {"action_realated": [],
                     "transition_related": [],
                     "useless": []}
        # Get the action related obs features
        # Get the next_obs related obs features
        # Get the useless obs features
        raw_config = deepcopy(self.raw_config)

        for obs_dim in range(self.obs_dims):
            if np.sum(causal_binary_matrix[obs_dim][self.obs_dims:self.obs_dims+self.action_dims]) > 0:
                obs_nodes["action_realated"].append(obs_dim)

        for obs_dim in range(self.obs_dims):
            if obs_dim in obs_nodes["action_realated"]:
                continue

            if np.sum(causal_binary_matrix[obs_dim][[_dim+self.obs_dims+self.action_dims for _dim in obs_nodes["action_realated"]]]) > 0:
                obs_nodes["transition_related"].append(obs_dim)
            else:
                obs_nodes["useless"].append(obs_dim)

        # TODO: Get the relation between
        # "action_realated" and "next_obs_related"

        # Generate decision graph
        data = deepcopy(self.data)
        obs = data.pop("obs")

        if len(obs_nodes["transition_related"]) > 0:
            # graph
            raw_config["metadata"]["graph"] = {'action': ['action_realated_obs'],

                                               'next_action_realated_obs': ['action_realated_obs',
                                                                            'transition_related_obs',
                                                                            'action'],

                                                'next_transition_related_obs': ['action_realated_obs',
                                                                                'transition_related_obs',
                                                                                'action']}
            # column
            obs_columns = [column for column in raw_config["metadata"]["columns"] if list(column.values())[0]["dim"] == "obs"]

            for obs_dim in obs_nodes["action_realated"]:
                obs_columns[obs_dim][list(obs_columns[obs_dim].keys())[0]]["dim"] = "action_realated_obs"
            for obs_dim in obs_nodes["transition_related"]:
                obs_columns[obs_dim][list(obs_columns[obs_dim].keys())[0]]["dim"] = "transition_related_obs"

            data["action_realated_obs"] = obs[:, obs_nodes["action_realated"]]
            data["transition_related_obs"] = obs[:, obs_nodes["transition_related"]]

        else:
            obs_columns = [column for column in raw_config["metadata"]["columns"] if list(column.values())[0]["dim"] == "obs"]
            for obs_dim in obs_nodes["action_realated"]:
                obs_columns[obs_dim][list(obs_columns[obs_dim].keys())[0]]["dim"] = "obs"
            data["obs"] = obs[:, obs_nodes["action_realated"]]
            # data["next_obs"] = next_obs[:,obs_nodes["action_realated"]]

            raw_config["metadata"]["graph"] = {'action': ['obs'],
                                               'next_obs': ['obs', 'action']}

        for obs_dim in obs_nodes["useless"]:
            obs_columns[obs_dim][list(obs_columns[obs_dim].keys())[0]]["dim"] = "useless_obs"

        with open(yaml_file, 'w') as f:
            yaml.dump(raw_config, f)

        np.savez_compressed(npz_file, **data)
