#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Callable, Iterable, Union

import torch

from torch import Tensor

from aac_metrics.classes.base import AACMetric
from aac_metrics.classes.bleu import BLEU
from aac_metrics.classes.cider_d import CIDErD
from aac_metrics.classes.fense import FENSE
from aac_metrics.classes.fluerr import FluErr
from aac_metrics.classes.meteor import METEOR
from aac_metrics.classes.rouge_l import ROUGEL
from aac_metrics.classes.sbert_sim import SBERTSim
from aac_metrics.classes.spice import SPICE
from aac_metrics.classes.spider import SPIDEr
from aac_metrics.classes.spider_fl import SPIDErFL
from aac_metrics.functional.evaluate import METRICS_SETS, evaluate


pylog = logging.getLogger(__name__)


class Evaluate(list[AACMetric], AACMetric):
    """Evaluate candidates with multiple references with custom metrics.

    For more information, see :func:`~aac_metrics.functional.evaluate.evaluate`.
    """

    full_state_update = False
    higher_is_better = None
    is_differentiable = False

    def __init__(
        self,
        preprocess: bool = True,
        metrics: Union[str, Iterable[str], Iterable[AACMetric]] = "aac",
        cache_path: str = "$HOME/.cache",
        java_path: str = "java",
        tmp_path: str = "/tmp",
        device: Union[str, torch.device, None] = "auto",
        verbose: int = 0,
    ) -> None:
        metrics = _instantiate_metrics_classes(
            metrics,
            cache_path,
            java_path,
            tmp_path,
            device,
            verbose,
        )

        list.__init__(self, metrics)
        AACMetric.__init__(self)
        self._preprocess = preprocess
        self._cache_path = cache_path
        self._java_path = java_path
        self._tmp_path = tmp_path
        self._device = device
        self._verbose = verbose

        self._candidates = []
        self._mult_references = []

    def compute(self) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        return evaluate(
            self._candidates,
            self._mult_references,
            self._preprocess,
            self,
            self._cache_path,
            self._java_path,
            self._tmp_path,
            self._device,
            self._verbose,
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
            "aac",
            cache_path,
            java_path,
            tmp_path,
            "auto",
            verbose,
        )


def _instantiate_metrics_classes(
    metrics: Union[str, Iterable[str], Iterable[AACMetric]] = "aac",
    cache_path: str = "$HOME/.cache",
    java_path: str = "java",
    tmp_path: str = "/tmp",
    device: Union[str, torch.device, None] = "auto",
    verbose: int = 0,
) -> list[AACMetric]:
    if isinstance(metrics, str) and metrics in METRICS_SETS:
        metrics = METRICS_SETS[metrics]

    if isinstance(metrics, str):
        metrics = [metrics]
    else:
        metrics = list(metrics)  # type: ignore

    metric_factory = _get_metric_factory_classes(
        True,
        cache_path,
        java_path,
        tmp_path,
        device,
        verbose,
    )

    metrics_inst: list[AACMetric] = []
    for metric in metrics:
        if isinstance(metric, str):
            metric = metric_factory[metric]()
        metrics_inst.append(metric)
    return metrics_inst


def _get_metric_factory_classes(
    return_all_scores: bool = True,
    cache_path: str = "$HOME/.cache",
    java_path: str = "java",
    tmp_path: str = "/tmp",
    device: Union[str, torch.device, None] = "auto",
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
        "cider_d": lambda: CIDErD(
            return_all_scores=return_all_scores,
        ),
        "spice": lambda: SPICE(
            return_all_scores=return_all_scores,
            cache_path=cache_path,
            java_path=java_path,
            tmp_path=tmp_path,
            verbose=verbose,
        ),
        "spider": lambda: SPIDEr(
            return_all_scores=return_all_scores,
            cache_path=cache_path,
            java_path=java_path,
            tmp_path=tmp_path,
            verbose=verbose,
        ),
        "sbert": lambda: SBERTSim(
            return_all_scores=return_all_scores,
            device=device,
            verbose=verbose,
        ),
        "fluerr": lambda: FluErr(
            return_all_scores=return_all_scores,
            device=device,
            verbose=verbose,
        ),
        "fense": lambda: FENSE(
            return_all_scores=return_all_scores,
            device=device,
            verbose=verbose,
        ),
        "spider_fl": lambda: SPIDErFL(
            return_all_scores=return_all_scores,
            cache_path=cache_path,
            java_path=java_path,
            tmp_path=tmp_path,
            device=device,
            verbose=verbose,
        ),
    }
