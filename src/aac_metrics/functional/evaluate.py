#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Iterable, Union

from torch import Tensor

from aac_metrics.classes.base import Metric
from aac_metrics.classes.coco_bleu import CocoBLEU
from aac_metrics.classes.coco_meteor import CocoMETEOR
from aac_metrics.classes.coco_rouge_l import CocoRougeL
from aac_metrics.classes.spider import SPIDEr
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
    use_ptb_tokenizer: bool = True,
    metrics: Union[str, Iterable[Metric]] = "aac",
    java_path: str = "java",
    tmp_path: str = "/tmp",
    cache_path: str = "$HOME/aac-metrics-cache",
    verbose: int = 0,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Evaluate candidates with multiple references with custom metrics.

    :param candidates: The list of N sentences.
    :param mult_references: The list of N lists of references.
    :param use_ptb_tokenizer: If True, the candidates and references wiill be passed as input to the PTB stanford tokenizer before computing metrics. defaults to True.
    :param metrics: The name of the metric list or the explicit list of metrics to compute. defaults to "aac".
    :param **kwargs: The keywords arguments passed to build the metrics.
    :returns: A tuple of 2 dictionaries containing respectively the global and local scores.
    """
    if isinstance(metrics, str):
        metrics = _get_metrics_list(
            metrics,
            java_path=java_path,
            tmp_path=tmp_path,
            cache_path=cache_path,
            verbose=verbose,
        )

    global_outs = {}
    local_outs = {}

    if use_ptb_tokenizer:
        candidates = preprocess_mono_sents(
            candidates, java_path, cache_path, tmp_path, verbose
        )
        mult_references = preprocess_mult_sents(
            mult_references, java_path, cache_path, tmp_path, verbose
        )

    for metric in metrics:
        global_outs_i, local_outs_i = metric(candidates, mult_references)
        global_outs |= global_outs_i
        local_outs |= local_outs_i

    return global_outs, local_outs


def aac_evaluate(
    candidates: list[str],
    mult_references: list[list[str]],
    use_ptb_tokenizer: bool = True,
    java_path: str = "java",
    tmp_path: str = "/tmp",
    cache_path: str = "$HOME/aac-metrics-cache",
    verbose: int = 0,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Evaluate candidates with multiple references with all Audio Captioing metrics.

    :param candidates: The list of N sentences.
    :param mult_references: The list of N lists of references.
    :param use_ptb_tokenizer: If True, the candidates and references wiill be passed as input to the PTB stanford tokenizer before computing metrics. defaults to True.
    :param java_path: The path to the java executable file. defaults to "java".
    :param tmp_path: The path to the temp directory. defaults to "/tmp".
    :param cache_path: The path to the aac-metrics cache directory. defaults to "$HOME/aac-metrics-cache".
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple of 2 dictionaries containing respectively the global and local scores.
    """
    return custom_evaluate(
        candidates,
        mult_references,
        use_ptb_tokenizer,
        metrics="aac",
        java_path=java_path,
        tmp_path=tmp_path,
        cache_path=cache_path,
        verbose=verbose,
    )


def _get_metrics_list(
    metric_set_name: str,
    java_path: str = "java",
    tmp_path: str = "/tmp",
    cache_path: str = "$HOME/aac-metrics-cache",
    verbose: int = 0,
) -> list[Metric]:
    metrics_factory = _get_metrics_factory(java_path, tmp_path, cache_path, verbose)

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
    java_path: str = "java",
    tmp_path: str = "/tmp",
    cache_path: str = "$HOME/aac-metrics-cache",
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
        "spider": lambda: SPIDEr(
            return_all_scores=True,
            java_path=java_path,
            tmp_path=tmp_path,
            cache_path=cache_path,
            verbose=verbose,
        ),
    }
