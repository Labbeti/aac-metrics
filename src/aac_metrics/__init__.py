#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Metrics for evaluating Automated Audio Captioning systems, designed for PyTorch.
"""


from .classes.base import AACMetric
from .classes.bleu import BLEU
from .classes.cider_d import CIDErD
from .classes.evaluate import DCASE2023Evaluate, _get_metric_factory_classes
from .classes.fense import FENSE
from .classes.meteor import METEOR
from .classes.rouge_l import ROUGEL
from .classes.spice import SPICE
from .classes.spider import SPIDEr
from .functional.evaluate import dcase2023_evaluate, evaluate
from .utils.paths import (
    get_default_cache_path,
    get_default_java_path,
    get_default_tmp_path,
    set_default_cache_path,
    set_default_java_path,
    set_default_tmp_path,
)
from . import download, eval, info


__name__ = "aac-metrics"
__author__ = "Etienne Labbé (Labbeti)"
__author_email__ = "labbeti.pub@gmail.com"
__license__ = "MIT"
__maintainer__ = "Etienne Labbé (Labbeti)"
__status__ = "Development"
__version__ = "0.4.5"


def load_metric(name: str, **kwargs) -> AACMetric:
    """Load a metric class by name.

    :param name: The name of the metric.
        Must be one of ("bleu_1", "bleu_2", "bleu_3", "bleu_4", "meteor", "rouge_l", "cider_d", "spice", "spider", "fense").
    :param **kwargs: The keyword optional arguments passed to the metric factory.
    :returns: The Metric object built.
    """
    name = name.lower().strip()

    factory = _get_metric_factory_classes(**kwargs)
    if name not in factory:
        raise ValueError(
            f"Invalid argument {name=}. (expected one of {tuple(factory.keys())})"
        )

    metric = factory[name]()
    return metric
