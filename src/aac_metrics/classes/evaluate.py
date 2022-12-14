#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Callable, Iterable, Union

from torch import Tensor

from aac_metrics.classes.base import AACMetric
from aac_metrics.classes.bleu import BLEU
from aac_metrics.classes.meteor import METEOR
from aac_metrics.classes.rouge_l import ROUGEL
from aac_metrics.classes.fense import FENSE
from aac_metrics.classes.spider import SPIDEr
from aac_metrics.functional.evaluate import METRICS_SETS, evaluate


logger = logging.getLogger(__name__)


class Evaluate(AACMetric, list[AACMetric]):
    """Evaluate candidates with multiple references with custom metrics.

    For more information, see :func:`~aac_metrics.functional.evaluate.custom_evaluate`.
    """

    full_state_update = False
    higher_is_better = None
    is_differentiable = False

    def __init__(
        self,
        preprocess: bool = True,
        cache_path: str = "$HOME/.cache",
        java_path: str = "java",
        tmp_path: str = "/tmp",
        verbose: int = 0,
        metrics: Union[str, Iterable[AACMetric]] = "aac",
    ) -> None:
        if isinstance(metrics, str):
            metrics = _get_metrics_classes_list(
                metrics,
                True,
                cache_path,
                java_path,
                tmp_path,
                verbose,
            )

        AACMetric.__init__(self)
        list.__init__(self, metrics)
        self._preprocess = preprocess
        self._cache_path = cache_path
        self._java_path = java_path
        self._tmp_path = tmp_path
        self._verbose = verbose

        self._candidates = []
        self._mult_references = []

    def compute(self) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        return evaluate(
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


class AACEvaluate(Evaluate):
    """Evaluate candidates with multiple references with all Audio Captioning metrics.

    For more information, see :func:`~aac_metrics.functional.evaluate.aac_evaluate`.
    """

    def __init__(
        self,
        preprocess: bool = True,
        cache_path: str = "$HOME/.cache",
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
    return_all_scores: bool = True,
    cache_path: str = "$HOME/.cache",
    java_path: str = "java",
    tmp_path: str = "/tmp",
    verbose: int = 0,
) -> list[AACMetric]:
    metrics_factory = _get_metrics_classes_factory(
        return_all_scores,
        cache_path,
        java_path,
        tmp_path,
        verbose,
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
    return_all_scores: bool = True,
    cache_path: str = "$HOME/.cache",
    java_path: str = "java",
    tmp_path: str = "/tmp",
    verbose: int = 0,
) -> dict[str, Callable[[], AACMetric]]:
    return {
        "bleu_1": lambda: BLEU(
            return_all_scores=return_all_scores,
            n=1,
        ),
        "bleu_2": lambda: BLEU(
            return_all_scores=return_all_scores,
            n=2,
        ),
        "bleu_3": lambda: BLEU(
            return_all_scores=return_all_scores,
            n=3,
        ),
        "bleu_4": lambda: BLEU(
            return_all_scores=return_all_scores,
            n=4,
        ),
        "meteor": lambda: METEOR(
            return_all_scores=return_all_scores,
            cache_path=cache_path,
            java_path=java_path,
            verbose=verbose,
        ),
        "rouge_l": lambda: ROUGEL(
            return_all_scores=return_all_scores,
        ),
        # Note: cider_d and spice and computed inside spider metric
        "spider": lambda: SPIDEr(
            return_all_scores=return_all_scores,
            cache_path=cache_path,
            java_path=java_path,
            tmp_path=tmp_path,
            verbose=verbose,
        ),
        "fense": lambda: FENSE(
            return_all_scores=return_all_scores,
            verbose=verbose,
        ),
    }
