#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import pickle
import zlib

from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

import torch

from torch import Tensor

from aac_metrics.classes.base import AACMetric
from aac_metrics.classes.bert_score_mrefs import BERTScoreMRefs
from aac_metrics.classes.bleu import BLEU, BLEU1, BLEU2, BLEU3, BLEU4
from aac_metrics.classes.cider_d import CIDErD
from aac_metrics.classes.fense import FENSE
from aac_metrics.classes.fer import FER
from aac_metrics.classes.meteor import METEOR
from aac_metrics.classes.rouge_l import ROUGEL
from aac_metrics.classes.sbert_sim import SBERTSim
from aac_metrics.classes.spice import SPICE
from aac_metrics.classes.spider import SPIDEr
from aac_metrics.classes.spider_max import SPIDErMax
from aac_metrics.classes.spider_fl import SPIDErFL
from aac_metrics.classes.vocab import Vocab
from aac_metrics.functional.evaluate import (
    DEFAULT_METRICS_SET_NAME,
    METRICS_SETS,
    evaluate,
)


pylog = logging.getLogger(__name__)


class Evaluate(list[AACMetric], AACMetric[tuple[dict[str, Tensor], dict[str, Tensor]]]):
    """Evaluate candidates with multiple references with custom metrics.

    For more information, see :func:`~aac_metrics.functional.evaluate.evaluate`.
    """

    full_state_update = False
    higher_is_better = None
    is_differentiable = False

    def __init__(
        self,
        preprocess: bool = True,
        metrics: Union[
            str, Iterable[str], Iterable[AACMetric]
        ] = DEFAULT_METRICS_SET_NAME,
        cache_path: Union[str, Path, None] = None,
        java_path: Union[str, Path, None] = None,
        tmp_path: Union[str, Path, None] = None,
        device: Union[str, torch.device, None] = "cuda_if_available",
        verbose: int = 0,
    ) -> None:
        metrics = _instantiate_metrics_classes(
            metrics=metrics,
            cache_path=cache_path,
            java_path=java_path,
            tmp_path=tmp_path,
            device=device,
            verbose=verbose,
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
            candidates=self._candidates,
            mult_references=self._mult_references,
            preprocess=self._preprocess,
            metrics=self,
            cache_path=self._cache_path,
            java_path=self._java_path,
            tmp_path=self._tmp_path,
            device=self._device,
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

    def tolist(self) -> list[AACMetric]:
        return list(self)

    def __hash__(self) -> int:
        # note: assume that all metrics can be pickled
        data = pickle.dumps(self)
        data = zlib.adler32(data)
        return data


class DCASE2023Evaluate(Evaluate):
    """Evaluate candidates with multiple references with DCASE2023 Audio Captioning metrics.

    For more information, see :func:`~aac_metrics.functional.evaluate.dcase2023_evaluate`.
    """

    def __init__(
        self,
        preprocess: bool = True,
        cache_path: Union[str, Path, None] = None,
        java_path: Union[str, Path, None] = None,
        tmp_path: Union[str, Path, None] = None,
        device: Union[str, torch.device, None] = "cuda_if_available",
        verbose: int = 0,
    ) -> None:
        super().__init__(
            preprocess=preprocess,
            metrics="dcase2023",
            cache_path=cache_path,
            java_path=java_path,
            tmp_path=tmp_path,
            device=device,
            verbose=verbose,
        )


def _instantiate_metrics_classes(
    metrics: Union[str, Iterable[str], Iterable[AACMetric]] = "aac",
    cache_path: Union[str, Path, None] = None,
    java_path: Union[str, Path, None] = None,
    tmp_path: Union[str, Path, None] = None,
    device: Union[str, torch.device, None] = "cuda_if_available",
    verbose: int = 0,
) -> list[AACMetric]:
    if isinstance(metrics, str) and metrics in METRICS_SETS:
        metrics = METRICS_SETS[metrics]

    if isinstance(metrics, str):
        metrics = [metrics]
    else:
        metrics = list(metrics)  # type: ignore

    metric_factory = _get_metric_factory_classes(
        return_all_scores=True,
        cache_path=cache_path,
        java_path=java_path,
        tmp_path=tmp_path,
        device=device,
        verbose=verbose,
    )

    metrics_inst: list[AACMetric] = []
    for metric in metrics:
        if isinstance(metric, str):
            metric = metric_factory[metric]()
        metrics_inst.append(metric)
    return metrics_inst


def _get_metric_factory_classes(
    return_all_scores: bool = True,
    cache_path: Union[str, Path, None] = None,
    java_path: Union[str, Path, None] = None,
    tmp_path: Union[str, Path, None] = None,
    device: Union[str, torch.device, None] = "cuda_if_available",
    verbose: int = 0,
    init_kwds: Optional[dict[str, Any]] = None,
) -> dict[str, Callable[[], AACMetric]]:
    if init_kwds is None or init_kwds is ...:
        init_kwds = {}

    init_kwds = init_kwds | dict(return_all_scores=return_all_scores)

    factory = {
        "bert_score": lambda: BERTScoreMRefs(
            verbose=verbose,
            **init_kwds,
        ),
        "bleu": lambda: BLEU(
            **init_kwds,
        ),
        "bleu_1": lambda: BLEU1(
            **init_kwds,
        ),
        "bleu_2": lambda: BLEU2(
            **init_kwds,
        ),
        "bleu_3": lambda: BLEU3(
            **init_kwds,
        ),
        "bleu_4": lambda: BLEU4(
            **init_kwds,
        ),
        "cider_d": lambda: CIDErD(
            **init_kwds,
        ),
        "fense": lambda: FENSE(
            device=device,
            verbose=verbose,
            **init_kwds,
        ),
        "fer": lambda: FER(
            device=device,
            verbose=verbose,
        ),
        "meteor": lambda: METEOR(
            cache_path=cache_path,
            java_path=java_path,
            verbose=verbose,
            **init_kwds,
        ),
        "rouge_l": lambda: ROUGEL(
            **init_kwds,
        ),
        "sbert_sim": lambda: SBERTSim(
            device=device,
            verbose=verbose,
            **init_kwds,
        ),
        "spice": lambda: SPICE(
            cache_path=cache_path,
            java_path=java_path,
            tmp_path=tmp_path,
            verbose=verbose,
            **init_kwds,
        ),
        "spider": lambda: SPIDEr(
            cache_path=cache_path,
            java_path=java_path,
            tmp_path=tmp_path,
            verbose=verbose,
            **init_kwds,
        ),
        "spider_fl": lambda: SPIDErFL(
            cache_path=cache_path,
            java_path=java_path,
            tmp_path=tmp_path,
            device=device,
            verbose=verbose,
            **init_kwds,
        ),
        "spider_max": lambda: SPIDErMax(
            cache_path=cache_path,
            java_path=java_path,
            tmp_path=tmp_path,
            verbose=verbose,
            **init_kwds,
        ),
        "vocab": lambda: Vocab(
            verbose=verbose,
            **init_kwds,
        ),
    }
    return factory
