from __future__ import annotations
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
from abc import ABC, abstractmethod
from operator import itemgetter
from typing import Any, Callable, Dict, Iterable, Optional, Union, Tuple, List
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
import time

import numpy as np

from causallearn.search.ConstraintBased.PC import pc as _pc
from causallearn.search.ConstraintBased.FCI import fci as _fci
from causallearn.search.ConstraintBased.CDNOD import cdnod as _cdnod
from causallearn.search.ScoreBased.GES import ges as _ges
from causallearn.search.ScoreBased.ExactSearch import bic_exact_search
from causallearn.search.FCMBased import lingam as _lingam
from causallearn.search.FCMBased.ANM.ANM import ANM

from causallearn.utils.cit import fisherz, chisq, gsq, mv_fisherz, kci, CIT
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge


# callback function: (current step, total step, current graph, whether use probability) -> Any
_CALLBACK_TYPE = Callable[[int, int, np.ndarray, bool], Any]


_CIT_METHODS = {
    "fisherz": fisherz,
    "chisq": chisq,
    "gsq": gsq,
    "mv_fisherz": mv_fisherz,
    "kci": kci
}


""" causal discovery methods """


# constraint-based
def pc(
    data: np.ndarray,
    indep: str = 'fisherz',
    thresh: float = 0.05,
    bg_rules: Optional[BackgroundKnowledge] = None,
    callback: Optional[_CALLBACK_TYPE] = None,
    **kwargs
) -> Tuple[np.ndarray, bool]:
    # start
    if callback:
        callback(0, 1, np.zeros((data.shape[-1], data.shape[-1])), False)

    cg = _pc(data, thresh, indep, True, 0, -1, background_knowledge=bg_rules)

    # end
    if callback:
        callback(1, 1, cg.G.graph, False)

    return cg.G.graph, False


def fci(
    data: np.ndarray,
    indep: str = 'fisherz',
    thresh: float = 0.05,
    bg_rules: Union[BackgroundKnowledge, None] = None,
    callback: Optional[_CALLBACK_TYPE] = None,
    **kwargs
) -> Tuple[np.ndarray, bool]:
    # start
    if callback:
        callback(0, 1, np.zeros((data.shape[-1], data.shape[-1])), False)

    G, _ = _fci(data, indep, thresh, verbose=False, background_knowledge=bg_rules)

    # end
    if callback:
        callback(1, 1, G.graph, False)

    return G.graph, False


def inter_cit(
    data: np.ndarray,
    indep: str = "fisherz",
    inter_classes: Iterable[Iterable[Iterable[int]]] = [],
    in_parallel: bool = True,
    parallel_limit: int = 5,
    callback: Optional[_CALLBACK_TYPE] = None,
    **kwargs
) -> Tuple[np.ndarray, bool]:
    """ use cit to discover the relations of variables
        inter different classes (indicated by indices)
    """
    n = data.shape[1]

    # initialize p-values graph and cit method
    p_values = np.zeros((n, n))
    cit = CIT(data, method=indep, **kwargs)
    # completed task counter
    completed = mp.Value("d", 0)

    # task parameters: cit algorithm, result matrix, input index, output index,
    # condition indices, callback, completed counter, total task number, (, lock)
    task_params = []
    for inter_pair in inter_classes:
        assert len(inter_pair) == 2, "Can only test relation between two classes"
        input_indices, output_indices = inter_pair[0], inter_pair[1]

        for idx in range(len(input_indices)):
            i = input_indices[idx]
            for o in output_indices:
                # params: cit algorithm, result matrix, input index,
                # output index, condition indices, completed counter
                task_params.append({
                    "cit": cit,
                    "mat": p_values,
                    "in_idx": i,
                    "out_idx": o,
                    "c_indices": input_indices[:idx]+input_indices[idx+1:],
                    "counter": completed,
                    "callback": callback,
                })
        # add total task number into the params
        for tp in task_params:
            tp["tot"] = len(task_params)

    # start
    if callback:
        callback(int(completed.value), len(task_params), p_values, True)

    # task function
    def do_cit_once(params):
        cit_alg, mat, x, y, C, cb, cnt, tot = itemgetter(
            "cit", "mat", "in_idx", "out_idx", "c_indices", "callback", "counter", "tot")(params)
        l = params["lock"] if "lock" in params else None

        mat[x, y] = 1 - cit_alg(x, y, condition_set=C)

        # modify number of completed tasks
        if l:
            l.acquire()
        cnt.value += 1
        if l:
            l.release()

        # callback
        if cb:
            callback(int(cnt.value), tot, mat, True)

    if in_parallel:
        # parallel running
        lock = mp.Lock()
        for tp in task_params:
            tp["lock"] = lock

        pool = ThreadPool(parallel_limit)
        pool.map(do_cit_once, task_params)
        pool.close()
        pool.join()
    else:
        # sequential running
        for i, tp in enumerate(task_params):
            do_cit_once(tp)

    return p_values, True


# FCM-based
def lingam(
    data: np.ndarray,
    ver: str = 'ica',
    callback: Optional[_CALLBACK_TYPE] = None,
    **kwargs
) -> Tuple[np.ndarray, bool]:
    model = _lingam.ICALiNGAM()

    if ver == 'direct':
        model = _lingam.DirectLiNGAM()
    elif ver == 'var':
        model = _lingam.VARLiNGAM()
    elif ver == 'rcd':
        model = _lingam.RCD()

    # start
    if callback:
        callback(0, 1, np.zeros(data.shape[-1], data.shape[-1]), True)

    model.fit(data)

    if ver == 'var':
        adj_matrix = model.adjacency_matrices_[0]
    else:
        adj_matrix = model.adjacency_matrix_

    # end
    if callback:
        callback(1, 1, np.abs(adj_matrix).T, True)

    return np.abs(adj_matrix).T, True


def anm(
    data: np.ndarray,
    kernelX: str = "Gaussian",
    kernelY: str = "Gaussian",
    callback: Optional[_CALLBACK_TYPE] = None,
    **kwargs
) -> Tuple[np.ndarray, bool]:
    model = ANM(kernelX, kernelY)

    graph = np.zeros((data.shape[-1], data.shape[-1]))

    # start
    if callback:
        callback(0, np.prod(graph.shape), graph, True)

    # orient edge by edge
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            p_value_f, p_value_b = model.cause_or_effect(data[:, i:i+1], data[:, j:j+1])
            graph[i, j], graph[j, i] = p_value_f, p_value_b

            # callback
            if callback:
                callback(i * graph.shape[1] + j + 1, np.prod(graph.shape), graph, True)

    return graph, True


# score-based
_AVAILABLE_SCORE_FNS = [
    "BIC", "BDeu", "CV_general", "marginal_general", "CV_multi", "marginal_multi",
]

def ges(
    data: np.ndarray,
    score_func: str = 'BIC',
    callback: Optional[_CALLBACK_TYPE] = None,
    **kwargs
) -> Tuple[np.ndarray, bool]:
    assert score_func in _AVAILABLE_SCORE_FNS, \
        "Do not support score function '{}' (available score functions: '{}')".format(
            score_func, "', '".join(_AVAILABLE_SCORE_FNS))
    load_func = f"local_score_{score_func}"

    # start
    if callback:
        callback(0, 1, np.zeros((data.shape[-1], data.shape[-1])), False)

    record = _ges(data, load_func)

    # end
    if callback:
        callback(1, 1, record['G'].graph, False)

    return record['G'].graph, False


_AVAILABLE_SEARCH_METHODS = [
    "astar", "dp",
]


def exact_search(
    data: np.ndarray,
    method: str = 'astar',
    callback: Optional[_CALLBACK_TYPE] = None,
    **kwargs
) -> Tuple[np.ndarray, bool]:
    assert method in _AVAILABLE_SEARCH_METHODS, \
        "Do not support search method '{}' (available search methods: '{}')".format(
            method, "', '".join(_AVAILABLE_SEARCH_METHODS))

    # start
    if callback:
        callback(0, 1, np.zeros((data.shape[-1], data.shape[-1])), False)

    dag, _ = bic_exact_search(data, search_method=method)

    # end
    if callback:
        callback(1, 1, dag, False)

    return dag, False


class Graph:
    """ Causal graph class """

    def __init__(
        self,
        graph: np.ndarray,
        is_real: bool = False,
        thresh_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        :param graph: ndarray, causal-learn style adjacency matrix,
            shape [state_dim + action_dim + state_dim, state_dim + action_dim + state_dim]
        :param is_real: whether the every element in graph is a real number
        :param thresh_info: information about the threshold of the graph, only used when the
            elements of the graph is real numbers
        """
        assert graph.shape[0] == graph.shape[1], "Graph required to be a square matrix"

        self._graph = graph.copy()
        self._is_real = is_real
        self._thresh_info = thresh_info

        self._adj_mat = self._format_graph(graph)

    def _format_graph(self, mat: np.ndarray) -> np.ndarray:
        """ format binary causal-learn style graph as adjacency matrix (DAG) """
        # causal-learn style binary matrix, -1 represents
        # the start of an edge, 1 represents the end of an edge
        mat = mat.copy()

        if not self._is_real:
            mat[np.arange(mat.shape[0]), np.arange(mat.shape[0])] = 0

            # i -> j
            single_direct = (mat == -1) & (mat.T == 1)

            # i - j or i <-> j
            bi_direct = ((mat == -1) & (mat.T == -1)) | ((mat == 1) & (mat == 1))

            # other area
            mat[(~single_direct) & (~bi_direct)] = 0

            # i -> j (m[i, j]=1, m[j, i]=0)
            mat[single_direct] = 1
            mat[single_direct.T] = 0

            # i - j or i <-> j (m[i, j] = m[j, i] = 1)
            mat[bi_direct] = 1
            mat[bi_direct.T] = 1

        return mat

    @property
    def graph(self):
        """ raw graph """
        return self._graph

    @property
    def thresh_info(self):
        """ information about threshold """
        return self._thresh_info

    def get_adj_matrix(self):
        """ return transition graph [S+A+S, S] (binary or real) """
        return self._adj_mat

    def get_binary_adj_matrix(self, thresh=None):
        """ return binary transition graph (with threshold specified) """
        if self._is_real:
            if thresh is None:
                thresh = 0.

            return (self._adj_mat > thresh).astype(int)
        else:
            return self._adj_mat

    def get_binary_adj_matrix_by_sparsity(self, sparsity=None):
        """ return binary transition graph (with sparsity specified) """
        if self._is_real:
            thresh = 0.
            if sparsity is not None:
                assert 0 <= sparsity and sparsity <= 1

                flatten_mat = self._adj_mat.reshape(-1)
                last_ele = int(np.floor(len(flatten_mat) * sparsity))
                last_ele = len(flatten_mat)-1 if last_ele >= len(flatten_mat) else last_ele
                arg_sorted = np.argsort(flatten_mat)
                thresh = flatten_mat[arg_sorted[last_ele]]

            return self.get_binary_adj_matrix(thresh)

        return self._adj_mat


class TransitionGraph(Graph):
    """ RL transition graph class """

    def __init__(
        self,
        graph: np.ndarray,
        state_dim: int,
        action_dim: int,
        is_real: bool = False,
        thresh_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        :param graph: ndarray, causal-learn style adjacency matrix,
            shape [state_dim + action_dim + state_dim, state_dim + action_dim + state_dim]
        :param state_dim: int, the dimension of state variables
        :param action_dim: int, the dimension of action variables
        :param is_real: whether the every element in graph is a real number
        :param thresh_info: information about the threshold of the graph, only used when the
            elements of the graph is real numbers
        """
        self._state_dim = state_dim
        self._action_dim = action_dim

        super().__init__(graph, is_real, thresh_info)

    def _format_graph(self, mat: np.ndarray) -> np.ndarray:
        """ format binary graph [S+A+S, S+A+S] as transition graph [S+A+S, S] """
        mat = mat.copy()

        if not self._is_real:
            # causal-learn style binary matrix, -1 represents
            # the start of an edge, 1 represents the end of an edge

            # state, action -> next_state
            inter_mat = mat[
                :self._state_dim+self._action_dim, self._state_dim+self._action_dim:]
            inter_mat[inter_mat == -1] = 1

            # state -> action
            inter_mat = mat[:self._state_dim, self._state_dim:self._state_dim+self._action_dim]
            inter_mat[inter_mat == -1] = 1

            # next_state -> next_state
            next_state_dim = mat.shape[0] - self._state_dim - self._action_dim
            for i in range(next_state_dim):
                start_idx = self._state_dim + self._action_dim + i
                # no loop
                mat[start_idx, start_idx] = 0
                for j in range(i + 1, next_state_dim):
                    end_idx = self._state_dim + self._action_dim + j

                    # start -> end
                    if mat[start_idx, end_idx] == -1 \
                        and mat[end_idx, start_idx] == 1:
                        mat[start_idx, end_idx] = 1
                        mat[end_idx, start_idx] = 0
                    # end -> start
                    elif mat[start_idx, end_idx] == 1 \
                        and mat[end_idx, start_idx] == -1:
                        mat[start_idx, end_idx] = 0
                        mat[end_idx, start_idx] = 1
                    # start - end
                    elif mat[start_idx, end_idx] \
                        == mat[end_idx, start_idx] == -1:
                        mat[start_idx, end_idx] = mat[end_idx, start_idx] = 1
                    # start <-> end
                    # TODO: handle unobserved confounder
                    elif mat[start_idx, end_idx] \
                        == mat[end_idx, start_idx] == 1:
                        mat[start_idx, end_idx] = mat[end_idx, start_idx] = 1

        # no ... -> state
        mat[:, :self._state_dim] = 0.
        # no action | next state -> action
        mat[self._state_dim:, self._state_dim:self._state_dim+self._action_dim] = 0.

        return mat


class DiscoveryModule(ABC):
    """ Base class for causal discovery modules """

    def __init__(self, **kwargs) -> None:
        self._graph: Union[Graph, None] = None

    @abstractmethod
    def fit(self, data: Any, **kwargs) -> DiscoveryModule:
        pass

    @property
    def graph(self) -> Union[Graph, None]:
        return self._graph


class ClassicalDiscovery(DiscoveryModule):
    """ Classical causal discovery algorithms """

    CLASSICAL_ALGOS = {
        "pc": pc,
        "fci": fci,
        "lingam": lingam,
        "anm": anm,
        "ges": ges,
        "exact_search": exact_search,
    }

    # designed only for transition graph
    CLASSICAL_ALGOS_TRANSITION = {
        "inter_cit": inter_cit,
    }

    CLASSICAL_ALGOS_THRESH_INFO = {
        "anm": {
            "min": 0., "max": 1., "common": 0.5,
        },
        "inter_cit": {
            "min": 0., "max": 1., "common": 0.8,
        },
        "lingam": {
            "min": 0., "max": float("inf"), "common": 0.01,
        }
    }

    def __init__(
        self,
        alg: str = "inter_cit",
        alg_args: Dict[str, Any] = {"indep": "kci", "in_parallel": False},
        state_keys: Optional[List[str]] = ["obs"],
        action_keys: Optional[List[str]] = ["action"],
        next_state_keys: Optional[List[str]] = ["next_obs"],
        limit: Optional[int] = 100,
        use_residual: bool = True,
        **kwargs
    ) -> None:
        """
        :param alg: str, algorithm name, options include
            'pc': Peter-Clark algorithm,
            'fci': Fast Causal Inference,
            'lingam': Linear Non-Gaussian Model,
            'anm': Additive Nonlinear Model,
            'ges': Greedy Equivalence Search,
            'exact_search': Exact Search,
        :param alg_args: additional arguments for the algorithm used,
            pc:
                indep: str, conditional independence test used, including
                    'fisherz': Fisher's Z conditional independence test,
                    'chisq': Chi-squared conditional independence test,
                    'gsq': G-squared conditional independence test,
                    'kci': Kernel-based conditional independence test,
                    'mv_fisherz': Missing-value Fisher's Z conditional independence test,
                thresh: float, level of significance for conditional independence test
            fci: the same as pc
            cit:
                indep: the same as 'indep' in pc,
                in_parallel: whether run the algorithm in parallel,
                parallel_limit: limit of the number of workers
            lingam:
                ver: the version of linear non-Gaussian model to use, including
                    'ica': ICA-based LiNGAM,
                    'direct': DirectLiNGAM,
                    'var': VAR-LiNGAM,
                    'rcd': RCD (repetitive causal discovery)
            anm:
                kernelX: the kernel function for cause data, including
                    'Gaussian': Gaussian kernel,
                    'Polynomial': Polynomial kernel,
                    'Linear': Linear kernel,
                kernelY: the kernel function for effect data, options are the same as kernelX
            ges:
                score_func: score function used to score the graph, including
                    'BIC': BIC score,
                    'BDeu': BDeu score,
                    'CV_general': Generalized score with cross validation
                        for data with single-dimensional variables
                    'marginal_general': Generalized score with marginal likelihood
                        for data with single-dimensional variables
                    'CV_multi': Generalized score with cross validation
                        for data with multi-dimensional variables
                    'marginal_multi': Generalized score with marginal likelihood
                        for data with multi-dimensional variables
            exact_search:
                method: the search method used, including:
                    'dp': dynamic programming (DP),
                    'astar': A* search,
        :param state_keys: list[str] | None, specifying the keys of
            states in the input data dictionary (None indicating not using transition data)
        :param action_keys: list[str] | None, specifying the keys of
            actions in the input data dictionary (None indicating not using transition data)
        :param next_state_keys: list[str] | None, specifying the keys of
            next states in the input data dictionary (None indicating not using transition data)
        :param limit: int | None, limit for the number of data samples used
        :param residual: bool, whether use residual as next states (only for transition data)
        """
        super().__init__(**kwargs)

        assert alg in ClassicalDiscovery.CLASSICAL_ALGOS \
            or alg in ClassicalDiscovery.CLASSICAL_ALGOS_TRANSITION, \
            "Do not support algorithm {} (available: '{}')".format(
                alg, "', '".join(
                    list(ClassicalDiscovery.CLASSICAL_ALGOS.keys()) + list(ClassicalDiscovery.CLASSICAL_ALGOS_TRANSITION.keys())
                ))
        assert (state_keys is not None and action_keys is not None and next_state_keys is not None) \
                or (state_keys is None and action_keys is None and next_state_keys is None), \
                "state, action, next states keys should all be None or not None"

        self._support_transition = state_keys is not None

        if self._support_transition:
            assert len(state_keys) > 0 and len(action_keys) > 0 and len(next_state_keys) > 0, \
                "state, action, next states keys can not be empty"

        self._alg_name = alg
        if alg in ClassicalDiscovery.CLASSICAL_ALGOS:
            self._alg = ClassicalDiscovery.CLASSICAL_ALGOS[alg]
        else:
            assert self._support_transition, f"{alg} only work for transition data"
            self._alg = ClassicalDiscovery.CLASSICAL_ALGOS_TRANSITION[alg]

        self._alg_args = alg_args
        self._state_keys = state_keys
        self._action_keys = action_keys
        self._next_state_keys = next_state_keys
        self._limit = limit
        self._use_residual = use_residual

    def _extract_data(
        self, data_dict: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        states = itemgetter(*self._state_keys)(data_dict)
        actions = itemgetter(*self._action_keys)(data_dict)
        next_states = itemgetter(*self._next_state_keys)(data_dict)

        if isinstance(states, tuple):
            states = np.concatenate(states, axis=-1)
        if isinstance(actions, tuple):
            actions = np.concatenate(actions, axis=-1)
        if isinstance(next_states, tuple):
            next_states = np.concatenate(next_states, axis=-1)

        return states, actions, next_states

    def _build_graph(
        self,
        graph_mat: np.ndarray,
        is_real: bool,
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
    ):
        """ build graph with graph adjacency matrix and whether the elements are real numbers
        :param graph_mat: ndarray, adjacency matrix of the graph
        :param is_real: whether the elements in the graph is real numbers
        :param state_dim: int or None, if not None, indicating the graph is a transition graph,
            data is shaped as [state_dim, action_dim, next_state_dim (state_dim)]
        :param action_dim: int or None, similar with state_dim
        """
        # information about the threshold of the edge existence (None if the graph is binary)
        thresh_info = None
        if self._alg_name in ClassicalDiscovery.CLASSICAL_ALGOS_THRESH_INFO:
            thresh_info = ClassicalDiscovery.CLASSICAL_ALGOS_THRESH_INFO[self._alg_name]

        if state_dim is not None and action_dim is not None:
            # build a transition graph
            self._graph = TransitionGraph(graph_mat, state_dim, action_dim,
                                          is_real, thresh_info)
        else:
            # build a regular graph
            self._graph = Graph(graph_mat, is_real, thresh_info)

    def _fit_transition(
        self,
        data: np.ndarray,
        state_dim: int,
        action_dim: int
    ) -> ClassicalDiscovery:
        """ fit the discovery module to transition data
        :param data: ndarray, transition data, aranged by states, actions, next_states
        :param state_dim: int, dimension of states
        :param action_dim: int, dimension of actions
        :return: the module itself
        """
        if self._alg_name in ClassicalDiscovery.CLASSICAL_ALGOS:
            # build general rl background knowledge rules
            obs_act_pattern = "^(%s)$" % "|".join(
                "X%d" % i for i in range(1, state_dim+action_dim+1))
            next_obs_pattern = "^(%s)$" % "|".join(
                "X%d" % i for i in range(state_dim+action_dim+1, data.shape[-1]+1))
            obs_pattern = "^(%s)$" % "|".join("X%d" % i for i in range(1, state_dim+1))
            act_pattern = "^(%s)$" % "|".join(
                "X%d" % i for i in range(state_dim+1, state_dim+action_dim+1))
            bg_rules = BackgroundKnowledge()
            # forbid next state -> state | action
            bg_rules.add_forbidden_by_pattern(next_obs_pattern,
                                              obs_act_pattern)
            # forbid action -> state
            bg_rules.add_forbidden_by_pattern(act_pattern, obs_pattern)

            # causal discovery algorithm, return a graph matrix
            # and whether the element of the matrix is real number
            graph, is_real = self._alg(data, bg_rules=bg_rules,
                                       **self._alg_args)
        else:
            # build general rl transition variable classes
            inter_classes = [
                # state -> action
                [
                    list(range(state_dim)),
                    list(range(state_dim, state_dim + action_dim))
                ],
                # state, action -> next state
                [
                    list(range(state_dim + action_dim)),
                    list(range(state_dim + action_dim, data.shape[1]))
                ]
            ]

            # causal discovery algorithm, return a graph matrix
            # and whether the element of the matrix is real number
            graph, is_real = self._alg(data, inter_classes=inter_classes, 
                                       **self._alg_args)

        # build transition graph
        self._build_graph(graph, is_real, state_dim, action_dim)

        return self

    def _fit_all(self, data: np.ndarray) -> ClassicalDiscovery:
        """ fit the discovery module to general data
        :param data: [num_samples x num_features]
        :return: the module itself
        """
        graph, is_real = self._alg(data, bg_rules=None, **self._alg_args)

        # build graph
        self._build_graph(graph, is_real)

        return self

    def fit(
        self,
        data: Union[Dict[str, np.ndarray], np.ndarray],
        fit_transition: bool = True,
    ) -> ClassicalDiscovery:
        """ fit the discovery module to transition data or general data
        :param data: dict[str, ndarray] | ndarray,
            transition data dictionary or general data matrix
        :return: the module itself
        """
        if fit_transition:
            assert isinstance(data, dict), "need transition data format"
            assert self._support_transition, \
                "fitting transition data needs specifying keys"

            states, actions, next_states = self._extract_data(data)
            state_dim, action_dim = states.shape[-1], actions.shape[-1]

            assert states.shape[0] == actions.shape[0] == next_states.shape[0], \
                "Transition data shape mismatch"

            # prepare data
            if self._use_residual:
                assert next_states.shape[1] == states.shape[1]
                next_states -= states

            data = np.concatenate([states, actions, next_states], axis=-1)

        else:
            assert isinstance(data, np.ndarray), "need general data format"
            assert self._alg_name not in ClassicalDiscovery.CLASSICAL_ALGOS_TRANSITION, \
                f"{self._alg_name} only work for transition data"

        # limited data samples
        limit = self._limit if self._limit is not None else data.shape[0]
        if limit <= data.shape[0]:
            indices = np.random.choice(data.shape[0],
                                       size=limit, replace=False)
            data = data[indices]

        # start fitting
        if fit_transition:
            return self._fit_transition(data, state_dim, action_dim)
        else:
            return self._fit_all(data)


class AsyncClassicalDiscovery(ClassicalDiscovery):
    """ Classical causal discovery algorithms (support asynchronous ver.) """

    def __init__(
        self,
        alg: str = "inter_cit",
        alg_args: Dict[str, Any] = { "indep": "kci","in_parallel": False },
        state_keys: Optional[List[str]] = ["obs"],
        action_keys: Optional[List[str]] = ["action"],
        next_state_keys: Optional[List[str]] = ["next_obs"],
        limit: Optional[int] = 100,
        use_residual: bool = True,
        callback: Optional[_CALLBACK_TYPE] = None,
        **kwargs
    ) -> None:
        super().__init__(
            alg, alg_args, state_keys, action_keys,
            next_state_keys, limit, use_residual, **kwargs)

        self._custom_callback = callback
        # init progress management
        self.cur_step = None
        self.tot_step = None
        self.start_time = None
        self.remaining_time = None
        self.is_running = False

    def _before_start(self):
        """ recording done before discovery starts """
        self.cur_step = 0
        self.tot_step = 0
        self.start_time = time.time()
        self.remaining_time = float("inf")
        self.is_running = True

    def _after_end(self):
        """ recording done after discovery ends """
        self.is_running = False

    def _callback(
        self,
        cur_step: int,
        tot_step: int,
        graph: np.ndarray,
        is_real: bool,
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
    ):
        # record progress
        self.cur_step = cur_step
        self.tot_step = tot_step
        self.elapsed_time = time.time() - self.start_time
        if self.cur_step != 0:
            self.remaining_time = self.elapsed_time * (tot_step / cur_step - 1)

        # record graph
        self._build_graph(graph, is_real, state_dim, action_dim)

        # custom callback
        if self._custom_callback:
            self._custom_callback(cur_step, tot_step, graph, is_real)

    def set_callback(self, callback: _CALLBACK_TYPE):
        """ set custom callback function """
        self._custom_callback = callback

    def fit(
        self,
        data: Union[Dict[str, np.ndarray], np.ndarray],
        fit_transition: bool = True
    ) -> ClassicalDiscovery:
        if fit_transition:
            assert isinstance(data, dict), "need transition data format"
            assert self._support_transition, \
                "fitting transition data needs specifying keys"
            states, actions, next_states = self._extract_data(data)
            state_dim, action_dim = states.shape[-1], actions.shape[-1]
            assert states.shape[0] == actions.shape[0] == next_states.shape[0], \
                "Transition data shape mismatch"

            # prepare data
            if self._use_residual:
                assert next_states.shape[1] == states.shape[1]
                next_states -= states
            data = np.concatenate([states, actions, next_states], axis=-1)

        else:
            assert isinstance(data, np.ndarray), "need general data format"
            assert self._alg_name not in AsyncClassicalDiscovery.CLASSICAL_ALGOS_TRANSITION, \
                f"{self._alg_name} only work for transition data"
            state_dim = action_dim = None

        # limited data samples
        limit = self._limit if self._limit is not None else data.shape[0]
        if limit <= data.shape[0]:
            indices = np.random.choice(data.shape[0], size=limit, replace=False)
            data = data[indices]

        # add callback into the algorithm arguments
        self._alg_args["callback"] = partial(
            self._callback, state_dim=state_dim, action_dim=action_dim)

        # start fitting
        self._before_start()

        try:
            if fit_transition:
                return self._fit_transition(data, state_dim, action_dim)
            else:
                return self._fit_all(data)
        finally:
            # finished
            self._after_end()


if __name__ == "__main__":
    np.random.seed(0)

    """ basic case for transition data """
    states = np.random.normal(0, 1, size=10000).reshape(-1, 2)
    actions = np.tanh(states[:, 0:1])
    delta_states = 2*states + actions + np.random.uniform(-0.1, 0.1, size=10000).reshape(-1, 2)
    next_states = delta_states + states
    # default keys
    data_dict = {"obs": states, "action": actions, "next_obs": next_states}

    # default keys are not None, the module supports transition data
    discover_module = ClassicalDiscovery()
    discover_module.fit(data_dict, fit_transition=True)
    # threshshold information (maybe None)
    print("graph threshold information\n", discover_module.graph.thresh_info)
    # for debug or private usage
    print("algorithm output graph\n", discover_module.graph.graph)
    # raw adjacent matrix
    print("transition graph\n", discover_module.graph.get_adj_matrix())
    # get binary matrix by indicating threshold (recommended)
    # 0.95 is a proper threshold for default method "inter_cit" (bounded in [0,1])
    print(
        "binary transition graph by threshold\n",
        discover_module.graph.get_binary_adj_matrix(discover_module.graph.thresh_info["common"])
    )
    # get binary matrix by indicating sparsity (need prior knowledge)
    print(
        "binary transition graph by sparsity\n",
        discover_module.graph.get_binary_adj_matrix_by_sparsity(0.75)
    )


    """ multi-keys """
    states2 = np.random.uniform(-1, 1, size=5000).reshape(-1, 1)
    delta_states2 = states2 - actions \
        + np.random.uniform(-0.1, 0.1, size=5000).reshape(-1, 1)
    next_states2 = delta_states2 + states2
    data_dict2 = {
        "obs1": states, "obs2": states2,
        "action": actions,
        "next_obs1": next_states, "next_obs2": next_states2
    }

    # keys are not None, the module supports transition data
    discover_module = ClassicalDiscovery(
        state_keys=["obs1", "obs2"],
        action_keys=["action"],
        next_state_keys=["next_obs1", "next_obs2"]
    )
    discover_module.fit(data_dict2, fit_transition=True)
    # threshshold information (maybe None)
    print("graph threshold information\n", discover_module.graph.thresh_info)
    # for debug or private usage
    print("algorithm output graph\n", discover_module.graph.graph)
    # raw adjacent matrix
    print("transition graph\n", discover_module.graph.get_adj_matrix())
    # get binary matrix by indicating threshold (recommended)
    # 0.95 is a proper threshold for default method "inter_cit" (bounded in [0,1])
    print(
        "binary transition graph by threshold\n",
        discover_module.graph.get_binary_adj_matrix(discover_module.graph.thresh_info["common"])
    )
    # get binary matrix by indicating sparsity (need prior knowledge)
    print(
        "binary transition graph by sparsity\n",
        discover_module.graph.get_binary_adj_matrix_by_sparsity(0.75)
    )


    """ basic case for general data (default algorithm does not support general data) """
    discover_module = ClassicalDiscovery(
        alg="lingam",
        alg_args={"ver": "direct"},
        limit=None,
    )
    s = np.random.uniform(-1, 1, 5000).reshape(-1, 1)
    a = 1.1 * s + np.random.uniform(-0.1, 0.1, size=5000).reshape(-1, 1)
    s_ = s + a + np.random.uniform(-0.01, 0.01, size=5000).reshape(-1, 1)
    data = np.concatenate((s, a, s_), axis=-1)

    discover_module.fit(data, fit_transition=False)
    # threshshold information (maybe None)
    print("graph threshold information\n", discover_module.graph.thresh_info)
    # for debug or private usage
    print("algorithm output graph\n", discover_module.graph.graph)
    # raw adjacent matrix
    print("transition graph\n", discover_module.graph.get_adj_matrix())
    # get binary matrix by indicating threshold (recommended)
    # 0.01 is a proper threshold for method "lingam" (>=0, not upper bounded)
    print(
        "binary transition graph by threshold\n",
        discover_module.graph.get_binary_adj_matrix(discover_module.graph.thresh_info["common"])
    )
    # get binary matrix by indicating sparsity (need prior knowledge)
    print(
        "binary transition graph by sparsity\n",
        discover_module.graph.get_binary_adj_matrix_by_sparsity(0.75)
    )


    """ use more data to acquire a more accurate result """
    discover_module = ClassicalDiscovery(
        # run in parallel
        alg_args={"indep": "kci", "in_parallel": True, "parallel_limit": 5},
        state_keys=["obs1", "obs2"],
        action_keys=["action"],
        next_state_keys=["next_obs1", "next_obs2"],
        limit=1000,
    )
    discover_module.fit(data_dict2, fit_transition=True)
    # threshshold information (maybe None)
    print("graph threshold information\n", discover_module.graph.thresh_info)
    # for debug or private usage
    print("algorithm output graph\n", discover_module.graph.graph)
    # raw adjacent matrix
    print("transition graph\n", discover_module.graph.get_adj_matrix())
    # get binary matrix by indicating threshold (recommended)
    # 0.95 is a proper threshold for default method "inter_cit" (bounded in [0,1])
    print(
        "binary transition graph by threshold\n",
        discover_module.graph.get_binary_adj_matrix(discover_module.graph.thresh_info["common"])
    )
    # get binary matrix by indicating sparsity (need prior knowledge)
    print(
        "binary transition graph by sparsity\n",
        discover_module.graph.get_binary_adj_matrix_by_sparsity(0.75)
    )


    """ use callback to informing progress """
    async_discover_module = AsyncClassicalDiscovery(
        state_keys=["obs1", "obs2"],
        action_keys=["action"],
        next_state_keys=["next_obs1", "next_obs2"],
        limit=3000,
        # callback can be set when constructing
        callback=None,
    )

    # callback type should be `_CALLBACK_TYPE`, see related comments
    def callback(cur_step: int, tot_step: int, cur_raw_graph: np.ndarray, is_real: bool):
        print(f"current step {cur_step}, total step {tot_step}, current raw graph {cur_raw_graph},"
            f" whether the elements of the graph is real number {is_real}")

        # can also get progress information from the module
        print(f"current step {async_discover_module.cur_step},"
            f" total step {async_discover_module.tot_step}")
        print(f"start time {async_discover_module.start_time}")
        print(f"time elapsed {async_discover_module.elapsed_time}")
        print(f"is running {async_discover_module.is_running}")
        print(f"estimated remaining time {async_discover_module.remaining_time}")
        print("currnet graph {}".format(
            async_discover_module.graph.get_binary_adj_matrix(
                async_discover_module.graph.thresh_info["common"])))

        print("=" * 20)

    # callback can also be set after constructed
    async_discover_module.set_callback(callback)

    # start fitting
    async_discover_module.fit(data_dict2, fit_transition=True)
    print(f"is running {async_discover_module.is_running}")
    # threshshold information (maybe None)
    print("graph threshold information\n", async_discover_module.graph.thresh_info)
    # for debug or private usage
    print("algorithm output graph\n", async_discover_module.graph.graph)
    # raw adjacent matrix
    print("transition graph\n", async_discover_module.graph.get_adj_matrix())
    # get binary matrix by indicating threshold (recommended)
    # 0.95 is a proper threshold for default method "inter_cit" (bounded in [0,1])
    print(
        "binary transition graph by threshold\n",
        async_discover_module.graph.get_binary_adj_matrix(
            async_discover_module.graph.thresh_info["common"])
    )
    # get binary matrix by indicating sparsity (need prior knowledge)
    print(
        "binary transition graph by sparsity\n",
        async_discover_module.graph.get_binary_adj_matrix_by_sparsity(0.75)
    )
