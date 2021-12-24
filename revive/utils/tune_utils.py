''''''
"""
    POLIXIR REVIVE, copyright (C) 2021 Polixir Technologies Co., Ltd., is 
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
import json
import torch
from typing import Dict
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
from ray.tune.logger import Logger, CSVLogger, JsonLogger, VALID_SUMMARY_TYPES
from ray.tune.utils import flatten_dict
from ray.tune.error import TuneError
from ray.tune import Stopper


class SysStopper(Stopper):
    def __init__(self, workspace, max_iter: int = 0):
        self._workspace = workspace
        self._max_iter = max_iter
        self._iter = defaultdict(lambda: 0)

    def __call__(self, trial_id, result):
        if self._max_iter > 0:
            self._iter[trial_id] += 1
            if self._iter[trial_id] >= self._max_iter:
                return True
        if result["stop_flag"]:
            return True
        
        return False

    def stop_all(self):
        if os.path.exists(os.path.join(self._workspace,'.env.json')):
            with open(os.path.join(self._workspace,'.env.json'), 'r') as f:
                _data = json.load(f)
            return _data["REVIVE_STOP"]
        else:
            return False

class TuneTBLogger(Logger):
    r"""
        custom tensorboard logger for ray tune
        modified from ray.tune.logger.TBXLogger
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

TUNE_LOGGERS = (CSVLogger, JsonLogger, TuneTBLogger)

import logging, copy
from ray.tune.suggest import BasicVariantGenerator, SearchGenerator, Searcher
from ray.tune.config_parser import create_trial_from_spec
from ray.tune.suggest.variant_generator import generate_variants, format_vars, resolve_nested_dict, flatten_resolved_vars
from ray.tune.trial import Trial
from ray.tune.utils import merge_dicts, flatten_dict
logger = logging.getLogger(__name__)

class CustomSearchGenerator(SearchGenerator):
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
        spec["config"] = merge_dicts(spec["config"],
                                     copy.deepcopy(suggested_config))

        # Create a new trial_id if duplicate trial is created
        flattened_config = resolve_nested_dict(spec["config"])
        self._counter += 1
        tag = "{0}_{1}".format(
            str(self._counter), format_vars(flattened_config))
        spec['config']['tag'] = tag # pass down the tag
        trial = create_trial_from_spec(
            spec,
            output_path,
            self._parser,
            evaluated_params=flatten_dict(suggested_config),
            experiment_tag=tag,
            trial_id=trial_id)
        return trial

class CustomBasicVariantGenerator(BasicVariantGenerator):
    def _generate_trials(self, num_samples, unresolved_spec, output_path=""):
        """Generates Trial objects with the variant generation process.

        Uses a fixed point iteration to resolve variants. All trials
        should be able to be generated at once.

        See also: `ray.tune.suggest.variant_generator`.

        Yields:
            Trial object
        """

        if "run" not in unresolved_spec:
            raise TuneError("Must specify `run` in {}".format(unresolved_spec))
        for _ in range(num_samples):
            for resolved_vars, spec in generate_variants(unresolved_spec):
                trial_id = self._uuid_prefix + ("%05d" % self._counter)
                self._counter += 1
                experiment_tag = str(self._counter)
                if resolved_vars:
                    experiment_tag += "_{}".format(format_vars(resolved_vars))
                spec['config']['tag'] = experiment_tag # pass down the tag
                yield create_trial_from_spec(
                    spec,
                    output_path,
                    self._parser,
                    evaluated_params=flatten_resolved_vars(resolved_vars),
                    trial_id=trial_id,
                    experiment_tag=experiment_tag)