#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Callable, Iterable, Union

from torch import Tensor

from aac_metrics.classes.base import Metric
from aac_metrics.classes.coco_bleu import CocoBLEU
from aac_metrics.classes.coco_meteor import CocoMETEOR
from aac_metrics.classes.coco_rouge_l import CocoRougeL
from aac_metrics.classes.spider import SPIDEr
from aac_metrics.functional.evaluate import METRICS_SETS, custom_evaluate


logger = logging.getLogger(__name__)


class CustomEvaluate(Metric, list[Metric]):
    """Evaluate candidates with multiple references with custom metrics.

    For more information, see :func:`~aac_metrics.functional.evaluate.custom_evaluate`.
    """

    full_state_update = False
    higher_is_better = None
    is_differentiable = False

    def __init__(
        self,
        preprocess: bool = True,
        cache_path: str = "$HOME/aac-metrics-cache",
        java_path: str = "java",
        tmp_path: str = "/tmp",
        verbose: int = 0,
        metrics: Union[str, Iterable[Metric]] = "all",
    ) -> None:
        if isinstance(metrics, str):
            metrics = _get_metrics_classes_list(
                metrics,
                cache_path=cache_path,
                java_path=java_path,
                tmp_path=tmp_path,
                verbose=verbose,
            )

        Metric.__init__(self)
        list.__init__(self, metrics)
        self._preprocess = preprocess
        self._cache_path = cache_path
        self._java_path = java_path
        self._tmp_path = tmp_path
        self._verbose = verbose

        self._candidates = []
        self._mult_references = []

    def compute(self) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        return custom_evaluate(
            self._candidates,
            self._mult_references,
            self._preprocess,
            self,
            cache_path=self._cache_path,
            java_path=self._java_path,
            tmp_path=self._tmp_path,
            verbose=self._verbose,
        )

    def reset(self) -> None:
        self._candidates = []
        self._mult_references = []
        return super().reset()

    def update(
        self,
        candidates: list[str],
        mult_references: list[list[str]],
    ) -> None:
        self._candidates += candidates
        self._mult_references += mult_references


class AACEvaluate(CustomEvaluate):
    """Evaluate candidates with multiple references with all Audio Captioning metrics.

    For more information, see :func:`~aac_metrics.functional.evaluate.aac_evaluate`.
    """

    def __init__(
        self,
        preprocess: bool = True,
        cache_path: str = "$HOME/aac-metrics-cache",
        java_path: str = "java",
        tmp_path: str = "/tmp",
        verbose: int = 0,
    ) -> None:
        super().__init__(
            preprocess,
            cache_path,
            java_path,
            tmp_path,
            verbose,
            "aac",
        )


def _get_metrics_classes_list(
    metric_set_name: str,
    cache_path: str = "$HOME/aac-metrics-cache",
    java_path: str = "java",
    tmp_path: str = "/tmp",
    verbose: int = 0,
) -> list[Metric]:
    metrics_factory = _get_metrics_classes_factory(
        cache_path, java_path, tmp_path, verbose
    )

    if metric_set_name in METRICS_SETS:
        metrics = [
            factory()
            for metric_name, factory in metrics_factory.items()
            if metric_name in METRICS_SETS[metric_set_name]
        ]
    else:
        raise ValueError(
            f"Invalid argument {metric_set_name=}. (expected one of {tuple(METRICS_SETS.keys())})"
        )

    return metrics


def _get_metrics_classes_factory(
    cache_path: str = "$HOME/aac-metrics-cache",
    java_path: str = "java",
    tmp_path: str = "/tmp",
    verbose: int = 0,
) -> dict[str, Callable[[], Metric]]:
    return {
        "bleu_1": lambda: CocoBLEU(
            return_all_scores=True,
            n=1,
        ),
        "bleu_2": lambda: CocoBLEU(
            return_all_scores=True,
            n=2,
        ),
        "bleu_3": lambda: CocoBLEU(
            return_all_scores=True,
            n=3,
        ),
        "bleu_4": lambda: CocoBLEU(
            return_all_scores=True,
            n=4,
        ),
        "meteor": lambda: CocoMETEOR(
            return_all_scores=True,
            cache_path=cache_path,
            java_path=java_path,
            verbose=verbose,
        ),
        "rouge_l": lambda: CocoRougeL(
            return_all_scores=True,
        ),
        # Note: cider_d and spice and computed inside spider metric
        "spider": lambda: SPIDEr(
            return_all_scores=True,
            cache_path=cache_path,
            java_path=java_path,
            tmp_path=tmp_path,
            verbose=verbose,
        ),
    }
