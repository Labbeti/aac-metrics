#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import time

from functools import partial
from typing import Any, Callable, Iterable, Union

import torch

from torch import Tensor

from aac_metrics.functional.bleu import bleu
from aac_metrics.functional.cider_d import cider_d
from aac_metrics.functional.fense import fense
from aac_metrics.functional.fluerr import fluerr
from aac_metrics.functional.meteor import meteor
from aac_metrics.functional.rouge_l import rouge_l
from aac_metrics.functional.sbert_sim import sbert_sim
from aac_metrics.functional.spice import spice
from aac_metrics.functional.spider import spider
from aac_metrics.functional.spider_fl import spider_fl
from aac_metrics.utils.checks import check_metric_inputs
from aac_metrics.utils.tokenization import preprocess_mono_sents, preprocess_mult_sents


pylog = logging.getLogger(__name__)


METRICS_SETS: dict[str, tuple[str, ...]] = {
    # Legacy metrics for AAC
    "default": (
        "bleu_1",
        "bleu_2",
        "bleu_3",
        "bleu_4",
        "meteor",
        "rouge_l",
        "spider",  # includes cider_d, spice
    ),
    # DCASE challenge task6a metrics for 2020, 2021 and 2022
    "dcase2020": (
        "bleu_1",
        "bleu_2",
        "bleu_3",
        "bleu_4",
        "meteor",
        "rouge_l",
        "spider",  # includes cider_d, spice
    ),
    # DCASE challenge task6a metrics for 2023
    "dcase2023": (
        "meteor",
        "spider_fl",  # includes cider_d, spice, spider, fluerr
    ),
    # All metrics
    "all": (
        "bleu_1",
        "bleu_2",
        "bleu_3",
        "bleu_4",
        "meteor",
        "rouge_l",
        "fense",  # includes sbert, fluerr
        "spider_fl",  # includes cider_d, spice, spider, fluerr
    ),
}
DEFAULT_METRICS_SET_NAME = "default"


def evaluate(
    candidates: list[str],
    mult_references: list[list[str]],
    preprocess: bool = True,
    metrics: Union[
        str, Iterable[str], Iterable[Callable[[list, list], tuple]]
    ] = DEFAULT_METRICS_SET_NAME,
    cache_path: str = ...,
    java_path: str = ...,
    tmp_path: str = ...,
    device: Union[str, torch.device, None] = "auto",
    verbose: int = 0,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Evaluate candidates with multiple references with custom metrics.

    :param candidates: The list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target.
    :param preprocess: If True, the candidates and references will be passed as input to the PTB stanford tokenizer before computing metrics.defaults to True.
    :param metrics: The name of the metric list or the explicit list of metrics to compute. defaults to "default".
    :param cache_path: The path to the external code directory. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_cache_path`.
    :param java_path: The path to the java executable. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_java_path`.
    :param tmp_path: Temporary directory path. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_tmp_path`.
    :param device: The PyTorch device used to run FENSE and SPIDErFL models.
        If None, it will try to detect use cuda if available. defaults to "auto".
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple of globals and locals scores.
    """
    check_metric_inputs(candidates, mult_references)

    metrics = _instantiate_metrics_functions(
        metrics, cache_path, java_path, tmp_path, device, verbose
    )

    if preprocess:
        common_kwds = dict(
            cache_path=cache_path,
            java_path=java_path,
            tmp_path=tmp_path,
            verbose=verbose,
        )
        candidates = preprocess_mono_sents(
            candidates,
            **common_kwds,
        )
        mult_references = preprocess_mult_sents(
            mult_references,
            **common_kwds,
        )

    outs_corpus = {}
    outs_sents = {}

    for i, metric in enumerate(metrics):
        if hasattr(metric, "__qualname__"):
            name = metric.__qualname__
        else:
            name = metric.__class__.__qualname__

        if verbose >= 1:
            pylog.info(f"[{i+1:2d}/{len(metrics):2d}] Computing {name} metric...")

        start = time.perf_counter()
        outs_corpus_i, outs_sents_i = metric(candidates, mult_references)
        end = time.perf_counter()

        if verbose >= 1:
            pylog.info(
                f"[{i+1:2d}/{len(metrics):2d}] Metric {name} computed in {end - start:.2f}s."
            )

        if __debug__:
            corpus_overlap = tuple(
                set(outs_corpus_i.keys()).intersection(outs_corpus.keys())
            )
            sents_overlap = tuple(
                set(outs_sents_i.keys()).intersection(outs_sents.keys())
            )
            if len(corpus_overlap) > 0 or len(sents_overlap) > 0:
                pylog.warning(
                    f"Found overlapping metric outputs names. (found {corpus_overlap=} and {sents_overlap=})"
                )

        outs_corpus |= outs_corpus_i
        outs_sents |= outs_sents_i

    return outs_corpus, outs_sents


def dcase2023_evaluate(
    candidates: list[str],
    mult_references: list[list[str]],
    preprocess: bool = True,
    cache_path: str = ...,
    java_path: str = ...,
    tmp_path: str = ...,
    device: Union[str, torch.device, None] = "auto",
    verbose: int = 0,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Evaluate candidates with multiple references with the DCASE2023 Audio Captioning metrics.

    :param candidates: The list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target.
    :param preprocess: If True, the candidates and references will be passed as input to the PTB stanford tokenizer before computing metrics.
        defaults to True.
    :param cache_path: The path to the external code directory. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_cache_path`.
    :param java_path: The path to the java executable. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_java_path`.
    :param tmp_path: Temporary directory path. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_tmp_path`.
    :param device: The PyTorch device used to run FENSE and SPIDErFL models.
        If None, it will try to detect use cuda if available. defaults to "auto".
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple of globals and locals scores.
    """
    return evaluate(
        candidates=candidates,
        mult_references=mult_references,
        preprocess=preprocess,
        metrics="dcase2023",
        cache_path=cache_path,
        java_path=java_path,
        tmp_path=tmp_path,
        device=device,
        verbose=verbose,
    )


def _instantiate_metrics_functions(
    metrics: Union[str, Iterable[str], Iterable[Callable[[list, list], tuple]]] = "all",
    cache_path: str = ...,
    java_path: str = ...,
    tmp_path: str = ...,
    device: Union[str, torch.device, None] = "auto",
    verbose: int = 0,
) -> list[Callable]:
    if isinstance(metrics, str) and metrics in METRICS_SETS:
        metrics = METRICS_SETS[metrics]

    if isinstance(metrics, str):
        metrics = [metrics]
    else:
        metrics = list(metrics)  # type: ignore

    if not all(isinstance(metric, (str, Callable)) for metric in metrics):
        raise TypeError(
            "Invalid argument type for metrics. (expected str, Iterable[str] or Iterable[Metric])"
        )

    metric_factory = _get_metric_factory_functions(
        return_all_scores=True,
        cache_path=cache_path,
        java_path=java_path,
        tmp_path=tmp_path,
        device=device,
        verbose=verbose,
    )

    metrics_inst: list[Callable] = []
    for metric in metrics:
        if isinstance(metric, str):
            metric = metric_factory[metric]
        metrics_inst.append(metric)
    return metrics_inst


def _get_metric_factory_functions(
    return_all_scores: bool = True,
    cache_path: str = ...,
    java_path: str = ...,
    tmp_path: str = ...,
    device: Union[str, torch.device, None] = "auto",
    verbose: int = 0,
) -> dict[str, Callable[[list[str], list[list[str]]], Any]]:
    return {
        "bleu": partial(
            bleu,
            return_all_scores=return_all_scores,
        ),
        "bleu_1": partial(
            bleu,
            return_all_scores=return_all_scores,
            n=1,
        ),
        "bleu_2": partial(
            bleu,
            return_all_scores=return_all_scores,
            n=2,
        ),
        "bleu_3": partial(
            bleu,
            return_all_scores=return_all_scores,
            n=3,
        ),
        "bleu_4": partial(
            bleu,
            return_all_scores=return_all_scores,
            n=4,
        ),
        "meteor": partial(
            meteor,
            return_all_scores=return_all_scores,
            cache_path=cache_path,
            java_path=java_path,
            verbose=verbose,
        ),
        "rouge_l": partial(
            rouge_l,
            return_all_scores=return_all_scores,
        ),
        "cider_d": partial(
            cider_d,
            return_all_scores=return_all_scores,
        ),
        "spice": partial(
            spice,
            return_all_scores=return_all_scores,
            cache_path=cache_path,
            java_path=java_path,
            tmp_path=tmp_path,
            verbose=verbose,
        ),
        "spider": partial(
            spider,
            return_all_scores=return_all_scores,
            cache_path=cache_path,
            java_path=java_path,
            tmp_path=tmp_path,
            verbose=verbose,
        ),
        "sbert_sim": partial(
            sbert_sim,
            return_all_scores=return_all_scores,
            device=device,
            verbose=verbose,
        ),
        "fluerr": partial(  # type: ignore
            fluerr,
            return_all_scores=return_all_scores,
            device=device,
            verbose=verbose,
        ),
        "fense": partial(
            fense,
            return_all_scores=return_all_scores,
            device=device,
            verbose=verbose,
        ),
        "spider_fl": partial(
            spider_fl,
            return_all_scores=return_all_scores,
            cache_path=cache_path,
            java_path=java_path,
            tmp_path=tmp_path,
            device=device,
            verbose=verbose,
        ),
    }
