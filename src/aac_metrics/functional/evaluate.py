#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import time

from functools import partial
from typing import Any, Callable, Iterable, Union

import torch

from torch import Tensor

from aac_metrics.functional.bleu import bleu
from aac_metrics.functional.fense import fense
from aac_metrics.functional.fluency_error import fluency_error
from aac_metrics.functional.meteor import meteor
from aac_metrics.functional.rouge_l import rouge_l
from aac_metrics.functional.sbert import sbert
from aac_metrics.functional.spider import spider
from aac_metrics.utils.tokenization import preprocess_mono_sents, preprocess_mult_sents


pylog = logging.getLogger(__name__)


METRICS_SETS = {
    "aac": (
        "bleu_1",
        "bleu_2",
        "bleu_3",
        "bleu_4",
        "meteor",
        "rouge_l",
        "cider_d",
        "spice",
        "spider",
    ),
    "all": (
        "bleu_1",
        "bleu_2",
        "bleu_3",
        "bleu_4",
        "meteor",
        "rouge_l",
        "cider_d",
        "spice",
        "spider",
        "fense",
    ),
}


def evaluate(
    candidates: list[str],
    mult_references: list[list[str]],
    preprocess: bool = True,
    metrics: Union[str, Iterable[Callable[[list, list], tuple]]] = "all",
    cache_path: str = "$HOME/.cache",
    java_path: str = "java",
    tmp_path: str = "/tmp",
    device: Union[str, torch.device, None] = "auto",
    verbose: int = 0,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Evaluate candidates with multiple references with custom metrics.

    :param candidates: The list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target.
    :param preprocess: If True, the candidates and references will be passed as input to the PTB stanford tokenizer before computing metrics.defaults to True.
    :param metrics: The name of the metric list or the explicit list of metrics to compute. defaults to "aac".
    :param cache_path: The path to the external code directory. defaults to "$HOME/.cache".
    :param java_path: The path to the java executable. defaults to "java".
    :param tmp_path: Temporary directory path. defaults to "/tmp".
    :param device: The PyTorch device used to run FENSE models. If None, it will try to detect use cuda if available. defaults to "cpu".
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple of globals and locals scores.
    """
    if isinstance(metrics, str):
        metrics = _get_metrics_functions_list(
            metrics,
            return_all_scores=True,
            cache_path=cache_path,
            java_path=java_path,
            tmp_path=tmp_path,
            device=device,
            verbose=verbose,
        )
    else:
        metrics = list(metrics)

    if preprocess:
        candidates = preprocess_mono_sents(
            candidates,
            cache_path,
            java_path,
            tmp_path,
            verbose=verbose,
        )
        mult_references = preprocess_mult_sents(
            mult_references,
            cache_path,
            java_path,
            tmp_path,
            verbose=verbose,
        )

    corpus_scores = {}
    sents_scores = {}

    for i, metric in enumerate(metrics):
        if hasattr(metric, "__qualname__"):
            name = metric.__qualname__
        else:
            name = metric.__class__.__qualname__

        if verbose >= 1:
            pylog.info(f"[{i+1:2d}/{len(metrics):2d}] Computing {name} metric...")
        start = time.perf_counter()

        corpus_scores_i, sents_scores_i = metric(candidates, mult_references)

        end = time.perf_counter()
        if verbose >= 1:
            pylog.info(
                f"[{i+1:2d}/{len(metrics):2d}] Metric {name} computed in {end - start:.2f}s."
            )

        corpus_scores |= corpus_scores_i
        sents_scores |= sents_scores_i

    return corpus_scores, sents_scores


def aac_evaluate(
    candidates: list[str],
    mult_references: list[list[str]],
    preprocess: bool = True,
    cache_path: str = "$HOME/.cache",
    java_path: str = "java",
    tmp_path: str = "/tmp",
    verbose: int = 0,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Evaluate candidates with multiple references with all Audio Captioning metrics.

    :param candidates: The list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target.
    :param preprocess: If True, the candidates and references will be passed as input to the PTB stanford tokenizer before computing metrics.
        defaults to True.
    :param cache_path: The path to the external code directory. defaults to "$HOME/.cache".
    :param java_path: The path to the java executable. defaults to "java".
    :param tmp_path: Temporary directory path. defaults to "/tmp".
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple of globals and locals scores.
    """
    return evaluate(
        candidates,
        mult_references,
        preprocess,
        "aac",
        cache_path,
        java_path,
        tmp_path,
        "cpu",
        verbose,
    )


def _get_metrics_functions_list(
    metric_set_name: str,
    return_all_scores: bool = True,
    cache_path: str = "$HOME/.cache",
    java_path: str = "java",
    tmp_path: str = "/tmp",
    device: Union[str, torch.device, None] = "auto",
    verbose: int = 0,
) -> list[Callable]:
    metrics_factory = _get_metrics_functions_factory(
        return_all_scores,
        cache_path,
        java_path,
        tmp_path,
        device,
        verbose,
    )

    if metric_set_name in METRICS_SETS:
        metrics = [
            factory
            for metric_name, factory in metrics_factory.items()
            if metric_name in METRICS_SETS[metric_set_name]
        ]
    else:
        raise ValueError(
            f"Invalid argument {metric_set_name=}. (expected one of {tuple(METRICS_SETS.keys())})"
        )

    return metrics


def _get_metrics_functions_factory(
    return_all_scores: bool = True,
    cache_path: str = "$HOME/.cache",
    java_path: str = "java",
    tmp_path: str = "/tmp",
    device: Union[str, torch.device, None] = "auto",
    verbose: int = 0,
) -> dict[str, Callable[[list[str], list[list[str]]], Any]]:
    return {
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
        # Note: cider_d and spice and computed inside spider metric
        "spider": partial(
            spider,
            return_all_scores=return_all_scores,
            cache_path=cache_path,
            java_path=java_path,
            tmp_path=tmp_path,
            verbose=verbose,
        ),
        "fense": partial(
            fense,
            return_all_scores=return_all_scores,
            device=device,
            verbose=verbose,
        ),
        "sbert": partial(
            sbert,
            return_all_scores=return_all_scores,
            device=device,
            verbose=verbose,
        ),
        "fluency_error": partial(
            fluency_error,
            return_all_scores=return_all_scores,
            device=device,
            verbose=verbose,
        ),
    }
