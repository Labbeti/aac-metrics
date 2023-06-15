#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Audio Captioning metrics package.
"""

__name__ = "aac-metrics"
__author__ = "Etienne Labbé (Labbeti)"
__author_email__ = "labbeti.pub@gmail.com"
__license__ = "MIT"
__maintainer__ = "Etienne Labbé (Labbeti)"
__status__ = "Development"
__version__ = "0.4.3"


from .classes.base import AACMetric
from .classes.bleu import BLEU
from .classes.cider_d import CIDErD
from .classes.evaluate import AACEvaluate, _get_metric_factory_classes
from .classes.fense import FENSE
from .classes.meteor import METEOR
from .classes.rouge_l import ROUGEL
from .classes.spice import SPICE
from .classes.spider import SPIDEr
from .functional.evaluate import dcase2023_evaluate, evaluate


__all__ = [
    "BLEU",
    "CIDErD",
    "AACEvaluate",
    "FENSE",
    "METEOR",
    "ROUGEL",
    "SPICE",
    "SPIDEr",
    "dcase2023_evaluate",
    "evaluate",
]


def load_metric(name: str, **kwargs) -> AACMetric:
    """Load a metric class by name.

    :param name: The name of the metric.
        Must be one of ("bleu_1", "bleu_2", "bleu_3", "bleu_4", "meteor", "rouge_l", "cider_d", "spice", "spider", "fense").
    :param **kwargs: The keyword optional arguments passed to the metric.
    :returns: The Metric object built.
    """
    name = name.lower().strip()

    factory = _get_metric_factory_classes(**kwargs)
    if name in factory:
        return factory[name]()
    else:
        raise ValueError(
            f"Invalid argument {name=}. (expected one of {tuple(factory.keys())})"
        )
