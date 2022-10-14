#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial
from typing import Any, Callable, Iterable, Union

from torch import Tensor

from aac_metrics.functional.coco_bleu import coco_bleu
from aac_metrics.functional.coco_meteor import coco_meteor
from aac_metrics.functional.coco_rouge_l import coco_rouge_l
from aac_metrics.functional.spider import spider
from aac_metrics.utils.tokenization import preprocess_mono_sents, preprocess_mult_sents


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
    ),
}


def custom_evaluate(
    candidates: list[str],
    mult_references: list[list[str]],
    preprocess: bool = True,
    metrics: Union[str, Iterable[Callable]] = "aac",
    cache_path: str = "$HOME/aac-metrics-cache",
    java_path: str = "java",
    tmp_path: str = "/tmp",
    verbose: int = 0,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Evaluate candidates with multiple references with custom metrics.

    :param candidates: The list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target.
    :param preprocess: If True, the candidates and references will be passed as input to the PTB stanford tokenizer before computing metrics.defaults to True.
    :param metrics: The name of the metric list or the explicit list of metrics to compute. defaults to "aac".
    :param cache_path: The path to the external code directory. defaults to "$HOME/aac-metrics-cache".
    :param java_path: The path to the java executable. defaults to "java".
    :param tmp_path: Temporary directory path. defaults to "/tmp".
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple of globals and locals scores.
    """
    if isinstance(metrics, str):
        metrics = _get_metrics_functions_list(
            metrics,
            cache_path=cache_path,
            java_path=java_path,
            tmp_path=tmp_path,
            verbose=verbose,
        )

    global_outs = {}
    local_outs = {}

    if preprocess:
        candidates = preprocess_mono_sents(
            candidates, cache_path, java_path, tmp_path, verbose
        )
        mult_references = preprocess_mult_sents(
            mult_references, cache_path, java_path, tmp_path, verbose
        )

    for metric in metrics:
        global_outs_i, local_outs_i = metric(candidates, mult_references)
        global_outs |= global_outs_i
        local_outs |= local_outs_i

    return global_outs, local_outs


def aac_evaluate(
    candidates: list[str],
    mult_references: list[list[str]],
    preprocess: bool = True,
    cache_path: str = "$HOME/aac-metrics-cache",
    java_path: str = "java",
    tmp_path: str = "/tmp",
    verbose: int = 0,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Evaluate candidates with multiple references with all Audio Captioning metrics.

    :param candidates: The list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target.
    :param preprocess: If True, the candidates and references will be passed as input to the PTB stanford tokenizer before computing metrics.
        defaults to True.
    :param cache_path: The path to the external code directory. defaults to "$HOME/aac-metrics-cache".
    :param java_path: The path to the java executable. defaults to "java".
    :param tmp_path: Temporary directory path. defaults to "/tmp".
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple of globals and locals scores.
    """
    return custom_evaluate(
        candidates,
        mult_references,
        preprocess,
        metrics="aac",
        cache_path=cache_path,
        java_path=java_path,
        tmp_path=tmp_path,
        verbose=verbose,
    )


def _get_metrics_functions_list(
    metric_set_name: str,
    cache_path: str = "$HOME/aac-metrics-cache",
    java_path: str = "java",
    tmp_path: str = "/tmp",
    verbose: int = 0,
) -> list[Callable]:
    metrics_factory = _get_metrics_functions_factory(
        cache_path, java_path, tmp_path, verbose
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
    cache_path: str = "$HOME/aac-metrics-cache",
    java_path: str = "java",
    tmp_path: str = "/tmp",
    verbose: int = 0,
) -> dict[str, Callable[[list[str], list[list[str]]], Any]]:
    return {
        "bleu_1": partial(
            coco_bleu,
            return_all_scores=True,
            n=1,
        ),
        "bleu_2": partial(
            coco_bleu,
            return_all_scores=True,
            n=2,
        ),
        "bleu_3": partial(
            coco_bleu,
            return_all_scores=True,
            n=3,
        ),
        "bleu_4": partial(
            coco_bleu,
            return_all_scores=True,
            n=4,
        ),
        "meteor": partial(
            coco_meteor,
            return_all_scores=True,
            cache_path=cache_path,
            java_path=java_path,
            verbose=verbose,
        ),
        "rouge_l": partial(
            coco_rouge_l,
            return_all_scores=True,
        ),
        # Note: cider_d and spice and computed inside spider metric
        "spider": partial(
            spider,
            return_all_scores=True,
            cache_path=cache_path,
            java_path=java_path,
            tmp_path=tmp_path,
            verbose=verbose,
        ),
    }
