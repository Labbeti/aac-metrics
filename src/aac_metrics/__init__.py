#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Metrics for evaluating Automated Audio Captioning systems, designed for PyTorch. """


__author__ = "Étienne Labbé (Labbeti)"
__author_email__ = "labbeti.pub@gmail.com"
__docs__ = "Audio Captioning Metrics"
__docs_url__ = "https://aac-metrics.readthedocs.io/en/stable/"
__license__ = "MIT"
__maintainer__ = "Étienne Labbé (Labbeti)"
__name__ = "aac-metrics"
__status__ = "Development"
__version__ = "0.5.5"


from .classes.base import AACMetric
from .classes.bert_score_mrefs import BERTScoreMRefs
from .classes.bleu import BLEU, BLEU1, BLEU2, BLEU3, BLEU4
from .classes.cider_d import CIDErD
from .classes.clap_sim import CLAPSim
from .classes.evaluate import (
    DCASE2023Evaluate,
    DCASE2024Evaluate,
    Evaluate,
    _get_metric_factory_classes,
    _instantiate_metrics_classes,
)
from .classes.fense import FENSE
from .classes.fer import FER
from .classes.mace import MACE
from .classes.meteor import METEOR
from .classes.rouge_l import ROUGEL
from .classes.sbert_sim import SBERTSim
from .classes.spice import SPICE
from .classes.spider import SPIDEr
from .classes.spider_fl import SPIDErFL
from .classes.spider_max import SPIDErMax
from .classes.vocab import Vocab
from .functional.evaluate import dcase2023_evaluate, dcase2024_evaluate, evaluate
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
    "BLEU1",
    "BLEU2",
    "BLEU3",
    "BLEU4",
    "CLAPSim",
    "CIDErD",
    "Evaluate",
    "DCASE2023Evaluate",
    "DCASE2024Evaluate",
    "FENSE",
    "FER",
    "MACE",
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
    "dcase2024_evaluate",
    "get_default_cache_path",
    "get_default_java_path",
    "get_default_tmp_path",
    "set_default_cache_path",
    "set_default_java_path",
    "set_default_tmp_path",
    "list_metrics_available",
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
    if not isinstance(name, str):
        raise TypeError(f"Invalid argument type {type(name)}. (expected str)")

    metrics = _instantiate_metrics_classes(name, **kwargs)
    return metrics[0]
