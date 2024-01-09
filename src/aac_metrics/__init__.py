#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Metrics for evaluating Automated Audio Captioning systems, designed for PyTorch. """


__author__ = "Etienne Labbé (Labbeti)"
__author_email__ = "labbeti.pub@gmail.com"
__license__ = "MIT"
__maintainer__ = "Etienne Labbé (Labbeti)"
__name__ = "aac-metrics"
__status__ = "Development"
__version__ = "0.5.3"


from .classes.base import AACMetric
from .classes.bert_score_mrefs import BERTScoreMRefs
from .classes.bleu import BLEU
from .classes.cider_d import CIDErD
from .classes.evaluate import Evaluate, DCASE2023Evaluate, _get_metric_factory_classes
from .classes.fer import FER
from .classes.fense import FENSE
from .classes.meteor import METEOR
from .classes.rouge_l import ROUGEL
from .classes.sbert_sim import SBERTSim
from .classes.spice import SPICE
from .classes.spider import SPIDEr
from .classes.spider_fl import SPIDErFL
from .classes.spider_max import SPIDErMax
from .classes.vocab import Vocab
from .functional.evaluate import evaluate, dcase2023_evaluate
from .utils.globals import (
    get_default_cache_path,
    get_default_java_path,
    get_default_tmp_path,
    set_default_cache_path,
    set_default_java_path,
    set_default_tmp_path,
)


__all__ = [
    "AACMetric",
    "BERTScoreMRefs",
    "BLEU",
    "CIDErD",
    "Evaluate",
    "DCASE2023Evaluate",
    "FENSE",
    "FER",
    "METEOR",
    "ROUGEL",
    "SBERTSim",
    "SPICE",
    "SPIDEr",
    "SPIDErFL",
    "SPIDErMax",
    "Vocab",
    "evaluate",
    "dcase2023_evaluate",
    "get_default_cache_path",
    "get_default_java_path",
    "get_default_tmp_path",
    "set_default_cache_path",
    "set_default_java_path",
    "set_default_tmp_path",
    "load_metric",
]


def list_metrics_available() -> list[str]:
    """Returns the list of metrics that can be loaded from its name."""
    factory = _get_metric_factory_classes()
    return list(factory.keys())


def load_metric(name: str, **kwargs) -> AACMetric:
    """Load a metric class by name.

    :param name: The name of the metric.
    :param **kwargs: The optional keyword arguments passed to the metric factory.
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
