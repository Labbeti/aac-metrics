#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import pickle
import zlib
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, Union

import torch
from torch import Tensor

from aac_metrics.classes.base import AACMetric
from aac_metrics.classes.bert_score_mrefs import BERTScoreMRefs
from aac_metrics.classes.bleu import BLEU, BLEU1, BLEU2, BLEU3, BLEU4
from aac_metrics.classes.cider_d import CIDErD
from aac_metrics.classes.clap_sim import CLAPSim
from aac_metrics.classes.fense import FENSE
from aac_metrics.classes.fer import FER
from aac_metrics.classes.mace import MACE
from aac_metrics.classes.meteor import METEOR
from aac_metrics.classes.rouge_l import ROUGEL
from aac_metrics.classes.sbert_sim import SBERTSim
from aac_metrics.classes.spice import SPICE
from aac_metrics.classes.spider import SPIDEr
from aac_metrics.classes.spider_fl import SPIDErFL
from aac_metrics.classes.spider_max import SPIDErMax
from aac_metrics.classes.vocab import Vocab
from aac_metrics.functional.evaluate import (
    DEFAULT_METRICS_SET_NAME,
    METRICS_SETS,
    evaluate,
    get_argnames,
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
        preprocess: Union[bool, Callable[[list[str]], list[str]]] = True,
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

    def __hash__(self) -> int:  # type: ignore
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


class DCASE2024Evaluate(Evaluate):
    """Evaluate candidates with multiple references with DCASE2024 Audio Captioning metrics.

    For more information, see :func:`~aac_metrics.functional.evaluate.dcase2024_evaluate`.
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
            metrics="dcase2024",
            cache_path=cache_path,
            java_path=java_path,
            tmp_path=tmp_path,
            device=device,
            verbose=verbose,
        )


def _instantiate_metrics_classes(
    metrics: Union[str, Iterable[str], Iterable[AACMetric]] = DEFAULT_METRICS_SET_NAME,
    *,
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


def _get_metric_factory_classes(**kwargs) -> dict[str, AACMetric]:
    classes: dict[str, type[AACMetric]] = {
        "bert_score": BERTScoreMRefs,
        "bleu": BLEU,
        "bleu_1": BLEU1,
        "bleu_2": BLEU2,
        "bleu_3": BLEU3,
        "bleu_4": BLEU4,
        "clap_sim": CLAPSim,
        "cider_d": CIDErD,
        "fer": FER,
        "fense": FENSE,
        "mace": MACE,
        "meteor": METEOR,
        "rouge_l": ROUGEL,
        "sbert_sim": SBERTSim,
        "spice": SPICE,
        "spider": SPIDEr,
        "spider_max": SPIDErMax,
        "spider_fl": SPIDErFL,
        "vocab": Vocab,
    }
    factory = {}
    for name, class_ in classes.items():
        argnames = get_argnames(class_)
        cls_kwargs = {k: v for k, v in kwargs.items() if k in argnames}
        metric = partial(class_, **cls_kwargs)
        factory[name] = metric
    return factory
