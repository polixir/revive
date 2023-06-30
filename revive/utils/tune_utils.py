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
import sys
import json
import torch
import numpy as np
import logging, copy
from ray.tune.experiment.trial import Trial
from ray.tune.utils import merge_dicts, flatten_dict
logger = logging.getLogger(__name__)
from typing import Dict, List
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from zoopt.parameter import ToolFunction
from ray.tune.logger import LoggerCallback, CSVLoggerCallback, JsonLoggerCallback
from ray.tune.utils import flatten_dict
from ray.tune.error import TuneError
from ray.tune import Stopper
from ray.tune import CLIReporter as _CLIReporter
from ray.tune.search.basic_variant import _flatten_resolved_vars, _count_spec_samples, _count_variants, _TrialIterator
from ray.tune.experiment import _convert_to_experiment_list
from ray.tune.search.basic_variant import warnings, Union, List, itertools, SERIALIZATION_THRESHOLD
from ray.tune.search.zoopt.zoopt_search import DEFAULT_METRIC, Solution, zoopt
from ray.tune.search.zoopt import ZOOptSearch as _ZOOptSearch
from ray.tune.search import BasicVariantGenerator, SearchGenerator, Searcher
from ray.tune.search.variant_generator import format_vars, _resolve_nested_dict, _flatten_resolved_vars
from ray.tune.experiment.config_parser import _create_trial_from_spec 


VALID_SUMMARY_TYPES = [int, float, np.float32, np.float64, np.int32, np.int64]

class SysStopper(Stopper):
    """Customizing the training mechanism of ray
    
    Reference : https://docs.ray.io/en/latest/tune/api/stoppers.html
    """
    def __init__(self, workspace, max_iter: int = 0, stop_callback = None):
        self._workspace = workspace
        self._max_iter = max_iter
        self._iter = defaultdict(lambda: 0)
        self.stop_callback = stop_callback

    # Customizing the stopping mechanism for a single trail
    def __call__(self, trial_id, result):
        if self._max_iter > 0:
            self._iter[trial_id] += 1
            if self._iter[trial_id] >= self._max_iter:
                return True
        if result["stop_flag"]:
            if self.stop_callback:
                self.stop_callback()
            return True
        
        return False

    # Customize the stopping mechanism for the entire training process
    def stop_all(self):
        if os.path.exists(os.path.join(self._workspace,'.env.json')):
            with open(os.path.join(self._workspace,'.env.json'), 'r') as f:
                _data = json.load(f)
            if _data["REVIVE_STOP"]:
                if self.stop_callback:
                    self.stop_callback()
            return _data["REVIVE_STOP"]
        else:
            return False

class TuneTBLoggerCallback(LoggerCallback):
    r"""
        custom tensorboard logger for ray tune
        modified from ray.tune.logger.TBXLogger
        
        Reference: https://docs.ray.io/en/latest/tune/api/doc/ray.tune.logger.LoggerCallback.html
    """
    def _init(self):
        self._file_writer = SummaryWriter(self.logdir)
        self.last_result = None
        self.step = 0

    def on_result(self, result):
        self.step += 1

        tmp = result.copy()
        flat_result = flatten_dict(tmp, delimiter="/")

        for k, v in flat_result.items():
            if type(v) in VALID_SUMMARY_TYPES:
                self._file_writer.add_scalar(k, float(v), global_step=self.step)
            elif isinstance(v, torch.Tensor):
                v = v.view(-1)
                self._file_writer.add_histogram(k, v, global_step=self.step)

        self.last_result = flat_result
        self.flush()

    def flush(self):
        if self._file_writer is not None:
            self._file_writer.flush()


def get_tune_callbacks():
    TUNELOGGERCallbacks = [CSVLoggerCallback, JsonLoggerCallback, TuneTBLoggerCallback]
    TUNELOGGERCallbacks = [callback() for callback in TUNELOGGERCallbacks] 

    return TUNELOGGERCallbacks


class CLIReporter(_CLIReporter):
    """Modifying the Command line reporter to support logging to loguru
    
    Reference : https://docs.ray.io/en/latest/tune/api/doc/ray.tune.CLIReporter.html
    
    """
    
    def report(self, trials: List, done: bool, *sys_info: Dict):
        message = self._progress_str(trials, done, *sys_info)
        from loguru import logger
        logger.info(f"{message}")

class CustomSearchGenerator(SearchGenerator):
    """
    Customize the SearchGenerator by placing tags in the spec's config

    Reference : https://github.com/ray-project/ray/blob/master/python/ray/tune/search/search_generator.py
    """
    def create_trial_if_possible(self, experiment_spec, output_path):
        logger.debug("creating trial")
        trial_id = Trial.generate_id()
        suggested_config = self.searcher.suggest(trial_id)

        if suggested_config == Searcher.FINISHED:
            self._finished = True
            logger.debug("Searcher has finished.")
            return

        if suggested_config is None:
            return

        spec = copy.deepcopy(experiment_spec)
        spec["config"] = merge_dicts(spec["config"],copy.deepcopy(suggested_config))

        # Create a new trial_id if duplicate trial is created
        flattened_config = _resolve_nested_dict(spec["config"])
        self._counter += 1
        tag = "{0}_{1}".format(str(self._counter), format_vars(flattened_config))
        spec['config']['tag'] = tag # pass down the tag
        trial = _create_trial_from_spec(
            spec,
            output_path,
            self._parser,
            evaluated_params=flatten_dict(suggested_config),
            experiment_tag=tag,
            trial_id=trial_id)
        return trial

class TrialIterator(_TrialIterator):
    """
    Customize the _TrialIterator by placing tags in the spec's config

    Reference : https://github.com/ray-project/ray/blob/master/python/ray/tune/search/basic_variant.py
    """
    def create_trial(self, resolved_vars, spec):
        trial_id = self.uuid_prefix + ("%05d" % self.counter)
        experiment_tag = str(self.counter)
        # Always append resolved vars to experiment tag?
        if resolved_vars:
            experiment_tag += "_{}".format(format_vars(resolved_vars))
        spec['config']['tag'] = experiment_tag
        self.counter += 1
        return _create_trial_from_spec(
            spec,
            self.output_path,
            self.parser,
            evaluated_params=_flatten_resolved_vars(resolved_vars),
            trial_id=trial_id,
            experiment_tag=experiment_tag)

class CustomBasicVariantGenerator(BasicVariantGenerator):
    """
    Using custom TrialIterator instead _TrialIterator
    
    Reference : https://github.com/ray-project/ray/blob/master/python/ray/tune/search/basic_variant.py
    """
    def add_configurations(
        self, experiments: Union["Experiment", List["Experiment"], Dict[str, Dict]]
    ):
        """Chains generator given experiment specifications.

        Arguments:
            experiments (Experiment | list | dict): Experiments to run.
        """
        experiment_list = _convert_to_experiment_list(experiments)
        for experiment in experiment_list:
            grid_vals = _count_spec_samples(experiment.spec, num_samples=1)
            lazy_eval = grid_vals > SERIALIZATION_THRESHOLD
            if lazy_eval:
                warnings.warn(
                    f"The number of pre-generated samples ({grid_vals}) "
                    "exceeds the serialization threshold "
                    f"({int(SERIALIZATION_THRESHOLD)}). Resume ability is "
                    "disabled. To fix this, reduce the number of "
                    "dimensions/size of the provided grid search.")

            previous_samples = self._total_samples
            points_to_evaluate = copy.deepcopy(self._points_to_evaluate)
            self._total_samples += _count_variants(experiment.spec,
                                                  points_to_evaluate)
            iterator = TrialIterator(
                uuid_prefix=self._uuid_prefix,
                num_samples=experiment.spec.get("num_samples", 1),
                unresolved_spec=experiment.spec,
                constant_grid_search=self._constant_grid_search,
                output_path=experiment.dir_name,
                points_to_evaluate=points_to_evaluate,
                lazy_eval=lazy_eval,
                start=previous_samples)
            self._iterators.append(iterator)
            self._trial_generator = itertools.chain(self._trial_generator,
                                                    iterator)
        

class Parameter(zoopt.Parameter):
    """
    Customize Zoom resource allocation method to fully utilize resources
    
    """
    def __init__(self, *args, **kwargs):
        self.parallel_num = kwargs.pop('parallel_num')
        super(Parameter, self).__init__(*args, **kwargs)
    
    def auto_set(self, budget):
        """
        Set train_size, positive_size, negative_size by following rules:
            budget < 3 --> error;
            budget < 3 --> train_size = p, positive_size = (0.2*self.parallel_num);

        :param budget: number of calls to the objective function
        :return: no return value
        """
        if budget < 3:
            ToolFunction.log('parameter.py: budget too small')
            sys.exit(1)
        else:
            if self.parallel_num < 4:
                super(Parameter, self).auto_set(budget)
                return 
            else:
                self.__train_size = self.parallel_num
                self.__positive_size = max(int(0.2 * self.parallel_num),1)
                self.__negative_size = self.__train_size - self.__positive_size


class ZOOptSearch(_ZOOptSearch):
    """
    Customize Zoom resource allocation method to fully utilize resources
    
    """
    def _setup_zoopt(self):
        if self._metric is None and self._mode:
            # If only a mode was passed, use anonymous metric
            self._metric = DEFAULT_METRIC

        _dim_list = []
        for k in self._dim_dict:
            self._dim_keys.append(k)
            _dim_list.append(self._dim_dict[k])

        init_samples = None
        if self._points_to_evaluate:
            logger.warning(
                "`points_to_evaluate` is ignored by ZOOpt in versions <= 0.4.1."
            )
            init_samples = [
                Solution(x=tuple(point[dim] for dim in self._dim_keys))
                for point in self._points_to_evaluate
            ]
        dim = zoopt.Dimension2(_dim_list)
        par = Parameter(budget=self._budget, init_samples=init_samples,parallel_num=self.parallel_num)
        if self._algo == "sracos" or self._algo == "asracos":
            from zoopt.algos.opt_algorithms.racos.sracos import SRacosTune
            self.optimizer = SRacosTune(
                dimension=dim,
                parameter=par,
                parallel_num=self.parallel_num,
                **self.kwargs
            )