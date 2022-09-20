#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Iterable, Union

from torch import Tensor

from aac_metrics.modules.base import Metric
from aac_metrics.modules.coco_bleu import CocoBLEU
from aac_metrics.modules.coco_meteor import CocoMETEOR
from aac_metrics.modules.coco_rouge_l import CocoRougeL
from aac_metrics.modules.coco_spider import CocoSPIDEr
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


def evaluate(
    candidates: list[str],
    mult_references: list[list[str]],
    use_ptb_tokenizer: bool = True,
    metrics: Union[str, Iterable[Callable]] = "aac",
    java_path: str = "java",
    tmp_path: str = "/tmp",
    cache_path: str = "$HOME/aac-metrics-cache",
    verbose: int = 0,
    *args,
    **kwargs,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    if isinstance(metrics, str):
        metrics = _get_metrics_set(
            metrics,
            *args,
            java_path=java_path,
            tmp_path=tmp_path,
            cache_path=cache_path,
            verbose=verbose,
            **kwargs,
        )

    global_outs = {}
    local_outs = {}

    if use_ptb_tokenizer:
        candidates = preprocess_mono_sents(candidates)
        mult_references = preprocess_mult_sents(mult_references)

    for metric in metrics:
        global_outs_i, local_outs_i = metric(candidates, mult_references)
        global_outs |= global_outs_i
        local_outs |= local_outs_i

    return global_outs, local_outs


def evaluate_mult_cands(
    mult_candidates: list[list[str]],
    mult_references: list[list[str]],
    metrics: Union[str, Iterable[Callable]] = "aac",
    *args,
    **kwargs,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    raise NotImplementedError


def _get_metrics_set(metric_set_name: str, **kwargs) -> list[Callable]:
    metrics_factory = _get_metrics_factory(**kwargs)

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


def _get_metrics_factory(
    java_path: str,
    tmp_path: str,
    cache_path: str,
    verbose: int = 0,
) -> dict[str, Callable[[], Metric]]:
    return {
        "bleu_1": lambda: CocoBLEU(True, 1),
        "bleu_2": lambda: CocoBLEU(True, 2),
        "bleu_3": lambda: CocoBLEU(True, 3),
        "bleu_4": lambda: CocoBLEU(True, 4),
        "meteor": lambda: CocoMETEOR(
            return_all_scores=True,
            java_path=java_path,
            cache_path=cache_path,
            verbose=verbose,
        ),
        "rouge_l": lambda: CocoRougeL(
            return_all_scores=True,
        ),
        # Note: cider_d and spice and computed inside spider metric
        "spider": lambda: CocoSPIDEr(
            return_all_scores=True,
            java_path=java_path,
            tmp_path=tmp_path,
            cache_path=cache_path,
            verbose=verbose,
        ),
    }
