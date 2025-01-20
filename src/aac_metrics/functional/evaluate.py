#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect
import logging
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterable, Union

import torch
from torch import Tensor, nn

from aac_metrics.functional.bert_score_mrefs import bert_score_mrefs
from aac_metrics.functional.bleu import bleu, bleu_1, bleu_2, bleu_3, bleu_4
from aac_metrics.functional.cider_d import cider_d
from aac_metrics.functional.clap_sim import clap_sim
from aac_metrics.functional.fense import fense
from aac_metrics.functional.fer import fer
from aac_metrics.functional.mace import mace
from aac_metrics.functional.meteor import meteor
from aac_metrics.functional.rouge_l import rouge_l
from aac_metrics.functional.sbert_sim import sbert_sim
from aac_metrics.functional.spice import spice
from aac_metrics.functional.spider import spider
from aac_metrics.functional.spider_fl import spider_fl
from aac_metrics.functional.spider_max import spider_max
from aac_metrics.functional.vocab import vocab
from aac_metrics.utils.checks import check_metric_inputs
from aac_metrics.utils.collections import flat_list_of_list, unflat_list_of_list
from aac_metrics.utils.log_utils import warn_once
from aac_metrics.utils.tokenization import preprocess_mono_sents

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
        "spider_fl",  # includes cider_d, spice, spider, fer
    ),
    # DCASE challenge task6 metrics for 2024
    "dcase2024": (
        "meteor",
        "spider_fl",  # includes cider_d, spice, spider, fer
        "fense",  # includes sbert, fer
        "vocab",
    ),
    # All metrics
    "all": (
        "bleu_1",
        "bleu_2",
        "bleu_3",
        "bleu_4",
        "meteor",
        "rouge_l",
        "fense",  # includes sbert, fer
        "spider_fl",  # includes cider_d, spice, spider, fer
        "vocab",
        "bert_score",
        "mace",
    ),
}
DEFAULT_METRICS_SET_NAME = "default"


def evaluate(
    candidates: list[str],
    mult_references: list[list[str]],
    preprocess: Union[bool, Callable[[list[str]], list[str]]] = True,
    metrics: Union[
        str, Iterable[str], Iterable[Callable[[list, list], tuple]]
    ] = DEFAULT_METRICS_SET_NAME,
    cache_path: Union[str, Path, None] = None,
    java_path: Union[str, Path, None] = None,
    tmp_path: Union[str, Path, None] = None,
    device: Union[str, torch.device, None] = "cuda_if_available",
    verbose: int = 0,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Evaluate candidates with multiple references with custom metrics.

    :param candidates: The list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target.
    :param preprocess: If True, the candidates and references will be passed as input to the PTB stanford tokenizer before computing metrics. defaults to True.
    :param metrics: The name of the metric list or the explicit list of metrics to compute. defaults to "default".
    :param cache_path: The path to the external code directory. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_cache_path`.
    :param java_path: The path to the java executable. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_java_path`.
    :param tmp_path: Temporary directory path. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_tmp_path`.
    :param device: The PyTorch device used to run FENSE and SPIDErFL models.
        If None, it will try to detect use cuda if available. defaults to "cuda_if_available".
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple contains the corpus and sentences scores.
    """
    check_metric_inputs(candidates, mult_references)

    metrics = _instantiate_metrics_functions(
        metrics,
        cache_path=cache_path,
        java_path=java_path,
        tmp_path=tmp_path,
        device=device,
        verbose=verbose,
    )

    # Note: we use == here because preprocess is not necessary a boolean
    if preprocess == False:  # noqa: E712
        preprocess = nn.Identity()

    elif preprocess == True:  # noqa: E712
        preprocess = partial(
            preprocess_mono_sents,
            cache_path=cache_path,
            java_path=java_path,
            tmp_path=tmp_path,
            verbose=verbose,
        )

    candidates = preprocess(candidates)
    mult_references_flat, sizes = flat_list_of_list(mult_references)
    mult_references_flat = preprocess(mult_references_flat)
    mult_references = unflat_list_of_list(mult_references_flat, sizes)

    outs_corpus = {}
    outs_sents = {}

    for i, metric in enumerate(metrics):
        if isinstance(metric, partial):
            name = metric.func.__qualname__
        elif hasattr(metric, "__qualname__"):
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
                warn_once(
                    f"Found overlapping metric outputs names. (found {corpus_overlap=} and {sents_overlap=} at least twice)",
                    pylog,
                )

        outs_corpus |= outs_corpus_i
        outs_sents |= outs_sents_i

    return outs_corpus, outs_sents


def dcase2023_evaluate(
    candidates: list[str],
    mult_references: list[list[str]],
    preprocess: Union[bool, Callable[[list[str]], list[str]]] = True,
    cache_path: Union[str, Path, None] = None,
    java_path: Union[str, Path, None] = None,
    tmp_path: Union[str, Path, None] = None,
    device: Union[str, torch.device, None] = "cuda_if_available",
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
        If None, it will try to detect use cuda if available. defaults to "cuda_if_available".
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple contains the corpus and sentences scores.
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


def dcase2024_evaluate(
    candidates: list[str],
    mult_references: list[list[str]],
    preprocess: Union[bool, Callable[[list[str]], list[str]]] = True,
    cache_path: Union[str, Path, None] = None,
    java_path: Union[str, Path, None] = None,
    tmp_path: Union[str, Path, None] = None,
    device: Union[str, torch.device, None] = "cuda_if_available",
    verbose: int = 0,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Evaluate candidates with multiple references with the DCASE2024 Audio Captioning metrics.

    :param candidates: The list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target.
    :param preprocess: If True, the candidates and references will be passed as input to the PTB stanford tokenizer before computing metrics.
        defaults to True.
    :param cache_path: The path to the external code directory. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_cache_path`.
    :param java_path: The path to the java executable. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_java_path`.
    :param tmp_path: Temporary directory path. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_tmp_path`.
    :param device: The PyTorch device used to run FENSE and SPIDErFL models.
        If None, it will try to detect use cuda if available. defaults to "cuda_if_available".
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple contains the corpus and sentences scores.
    """
    return evaluate(
        candidates=candidates,
        mult_references=mult_references,
        preprocess=preprocess,
        metrics="dcase2024",
        cache_path=cache_path,
        java_path=java_path,
        tmp_path=tmp_path,
        device=device,
        verbose=verbose,
    )


def _instantiate_metrics_functions(
    metrics: Union[str, Iterable[str], Iterable[Callable[[list, list], tuple]]] = "all",
    *,
    cache_path: Union[str, Path, None] = None,
    java_path: Union[str, Path, None] = None,
    tmp_path: Union[str, Path, None] = None,
    device: Union[str, torch.device, None] = "cuda_if_available",
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
    **kwargs,
) -> dict[str, Callable[[list[str], list[list[str]]], Any]]:
    functions = {
        "bert_score": bert_score_mrefs,
        "bleu": bleu,
        "bleu_1": bleu_1,
        "bleu_2": bleu_2,
        "bleu_3": bleu_3,
        "bleu_4": bleu_4,
        "clap_sim": clap_sim,
        "cider_d": cider_d,
        "fer": fer,
        "fense": fense,
        "mace": mace,
        "meteor": meteor,
        "rouge_l": rouge_l,
        "sbert_sim": sbert_sim,
        "spice": spice,
        "spider": spider,
        "spider_max": spider_max,
        "spider_fl": spider_fl,
        "vocab": vocab,
    }
    factory = {}
    for name, fn in functions.items():
        argnames = get_argnames(fn)
        fn_kwargs = {k: v for k, v in kwargs.items() if k in argnames}
        factory[name] = partial(fn, **fn_kwargs)
    return factory


def get_argnames(fn: Callable) -> list[str]:
    """Get arguments names of a method, function or callable object."""
    if inspect.ismethod(fn):
        # If method, remove 'self' arg
        argnames = fn.__code__.co_varnames[1:]  # type: ignore
    elif inspect.isfunction(fn):
        argnames = fn.__code__.co_varnames
    else:
        argnames = fn.__call__.__code__.co_varnames  # type: ignore

    argnames = list(argnames)
    return argnames
