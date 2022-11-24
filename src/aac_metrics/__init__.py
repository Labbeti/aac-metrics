#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Audio Captioning metrics.
"""

__name__ = "aac-metrics"
__author__ = "Etienne Labbé (Labbeti)"
__author_email__ = "labbeti.pub@gmail.com"
__license__ = "MIT"
__maintainer__ = "Etienne Labbé (Labbeti)"
__status__ = "Development"
__version__ = "0.1.2"


from .classes.base import AACMetric
from .classes.coco_bleu import CocoBLEU
from .classes.coco_cider_d import CocoCIDErD
from .classes.coco_meteor import CocoMETEOR
from .classes.coco_rouge_l import CocoRougeL
from .classes.coco_spice import CocoSPICE
from .classes.evaluate import AACEvaluate, _get_metrics_classes_factory
from .classes.fense import FENSE
from .classes.spider import SPIDEr
from .functional.evaluate import aac_evaluate


__all__ = [
    "CocoBLEU",
    "CocoCIDErD",
    "CocoMETEOR",
    "CocoRougeL",
    "CocoSPICE",
    "AACEvaluate",
    "FENSE",
    "SPIDEr",
    "aac_evaluate",
]


def load_metric(name: str, **kwargs) -> AACMetric:
    """Load a metric class by name.

    :param name: The name of the metric.
        Must be one of ("bleu_1", "bleu_2", "bleu_3", "bleu_4", "meteor", "rouge_l", "cider_d", "spice", "spider", "fense").
    :param **kwargs: The keyword optional arguments passed to the metric.
    :returns: The Metric object built.
    """
    name = name.lower().strip()

    factory = _get_metrics_classes_factory(**kwargs)
    if name in factory:
        return factory[name]()
    else:
        raise ValueError(
            f"Invalid argument {name=}. (expected one of {tuple(factory.keys())})"
        )
