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
import collections

BATCH_COUNT = "batch_count"
NUM_SAMPLES = "num_samples"
BATCH_SIZE = "*batch_size"


class AverageMeter:
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update current value, total sum, and average."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AverageMeterCollection:
    """
    This is a class called AverageMeterCollection that calculates and stores the average metrics for a collection of meters.
    """
    def __init__(self):
        self._batch_count = 0
        self.n = 0
        self._meters = collections.defaultdict(AverageMeter)

    def update(self, metrics, n=1):
        """Does one batch of updates for the provided metrics."""
        self._batch_count += 1
        self.n += n
        for metric, value in metrics.items():
            self._meters[metric].update(value, n=n)

    def summary(self):
        """Returns a dict of average and most recent values for each metric."""
        stats = {BATCH_COUNT: self._batch_count, NUM_SAMPLES: self.n}
        for metric, meter in self._meters.items():
            stats[str(metric)] = meter.avg
            stats["last_" + str(metric)] = meter.val
        return stats