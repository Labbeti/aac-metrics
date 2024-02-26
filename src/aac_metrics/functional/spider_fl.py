#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from pathlib import Path
from typing import Callable, Iterable, Optional, Union

import torch

from torch import Tensor
from transformers.models.auto.tokenization_auto import AutoTokenizer

from aac_metrics.functional.fer import (
    fer,
    _load_echecker_and_tokenizer,
    BERTFlatClassifier,
    DEFAULT_FER_MODEL,
)
from aac_metrics.functional.spider import spider
from aac_metrics.utils.checks import check_metric_inputs


pylog = logging.getLogger(__name__)


def spider_fl(
    candidates: list[str],
    mult_references: list[list[str]],
    return_all_scores: bool = True,
    # CIDErD args
    n: int = 4,
    sigma: float = 6.0,
    tokenizer: Callable[[str], list[str]] = str.split,
    return_tfidf: bool = False,
    # SPICE args
    cache_path: Union[str, Path, None] = None,
    java_path: Union[str, Path, None] = None,
    tmp_path: Union[str, Path, None] = None,
    n_threads: Optional[int] = None,
    java_max_memory: str = "8G",
    timeout: Union[None, int, Iterable[int]] = None,
    # FluencyError args
    echecker: Union[str, BERTFlatClassifier] = DEFAULT_FER_MODEL,
    echecker_tokenizer: Optional[AutoTokenizer] = None,
    error_threshold: float = 0.9,
    device: Union[str, torch.device, None] = "cuda_if_available",
    batch_size: int = 32,
    reset_state: bool = True,
    return_probs: bool = True,
    # Other args
    penalty: float = 0.9,
    verbose: int = 0,
) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Tensor]]]:
    """Combinaison of SPIDEr with Fluency Error detector.

    - Original implementation: https://github.com/felixgontier/dcase-2023-baseline/blob/main/metrics.py#L48.

    .. warning::
        This metric requires at least 2 candidates with 2 sets of references, otherwise it will raises a ValueError.

    :param candidates: The list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target.
    :param return_all_scores: If True, returns a tuple containing the globals and locals scores.
        Otherwise returns a scalar tensor containing the main global score.
        defaults to True.
    :param n: Maximal number of n-grams taken into account. defaults to 4.
    :param sigma: Standard deviation parameter used for gaussian penalty. defaults to 6.0.
    :param tokenizer: The fast tokenizer used to split sentences into words. defaults to str.split.
    :param return_tfidf: If True, returns the list of dictionaries containing the tf-idf scores of n-grams in the sents_score output.
        defaults to False.
    :param cache_path: The path to the external code directory. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_cache_path`.
    :param java_path: The path to the java executable. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_java_path`.
    :param tmp_path: Temporary directory path. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_tmp_path`.
    :param n_threads: Number of threads used to compute SPICE.
        None value will use the default value of the java program.
        defaults to None.
    :param java_max_memory: The maximal java memory used. defaults to "8G".
    :param timeout: The number of seconds before killing the java subprogram.
        If a list is given, it will restart the program if the i-th timeout is reached.
        If None, no timeout will be used.
        defaults to None.
    :param echecker: The echecker model used to detect fluency errors.
        Can be "echecker_clotho_audiocaps_base", "echecker_clotho_audiocaps_tiny", "none" or None.
        defaults to "echecker_clotho_audiocaps_base".
    :param echecker_tokenizer: The tokenizer of the echecker model.
        If None and echecker is not None, this value will be inferred with `echecker.model_type`.
        defaults to None.
    :param error_threshold: The threshold used to detect fluency errors for echecker model. defaults to 0.9.
    :param device: The PyTorch device used to run FENSE models. If "cuda_if_available", it will use cuda if available. defaults to "cuda_if_available".
    :param batch_size: The batch size of the sBERT and echecker models. defaults to 32.
    :param reset_state: If True, reset the state of the PyTorch global generator after the initialization of the pre-trained models. defaults to True.
    :param return_probs: If True, return each individual error probability given by the fluency detector model. defaults to True.
    :param penalty: The penalty coefficient applied. Higher value means to lower the cos-sim scores when an error is detected. defaults to 0.9.
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple of globals and locals scores or a scalar tensor with the main global score.
    """
    check_metric_inputs(candidates, mult_references)

    # Init models
    echecker, echecker_tokenizer = _load_echecker_and_tokenizer(
        echecker=echecker,
        echecker_tokenizer=echecker_tokenizer,
        device=device,
        reset_state=reset_state,
        verbose=verbose,
    )
    spider_outs: tuple[dict[str, Tensor], dict[str, Tensor]] = spider(  # type: ignore
        candidates=candidates,
        mult_references=mult_references,
        return_all_scores=True,
        n=n,
        sigma=sigma,
        tokenizer=tokenizer,
        return_tfidf=return_tfidf,
        cache_path=cache_path,
        java_path=java_path,
        tmp_path=tmp_path,
        n_threads=n_threads,
        java_max_memory=java_max_memory,
        timeout=timeout,
        verbose=verbose,
    )
    fer_outs: tuple[dict[str, Tensor], dict[str, Tensor]] = fer(  # type: ignore
        candidates=candidates,
        return_all_scores=True,
        echecker=echecker,
        echecker_tokenizer=echecker_tokenizer,
        error_threshold=error_threshold,
        device=device,
        batch_size=batch_size,
        reset_state=reset_state,
        return_probs=return_probs,
        verbose=verbose,
    )
    spider_fl_outs = _spider_fl_from_outputs(spider_outs, fer_outs, penalty)

    if return_all_scores:
        return spider_fl_outs
    else:
        return spider_fl_outs[0]["spider_fl"]


def _spider_fl_from_outputs(
    spider_outs: tuple[dict[str, Tensor], dict[str, Tensor]],
    fer_outs: tuple[dict[str, Tensor], dict[str, Tensor]],
    penalty: float = 0.9,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Combines SPIDEr and FER outputs.

    Based on https://github.com/felixgontier/dcase-2023-baseline/blob/main/metrics.py#L48
    """
    spider_outs_corpus, spider_outs_sents = spider_outs
    fer_outs_corpus, fer_outs_sents = fer_outs

    spider_scores = spider_outs_sents["spider"]
    fer_scores = fer_outs_sents["fer"]
    spider_fl_scores = spider_scores * (1.0 - penalty * fer_scores)
    spider_fl_score = spider_fl_scores.mean()

    spider_fl_outs_corpus = (
        spider_outs_corpus | fer_outs_corpus | {"spider_fl": spider_fl_score}
    )
    spider_fl_outs_sents = (
        spider_outs_sents | fer_outs_sents | {"spider_fl": spider_fl_scores}
    )
    spider_fl_outs = spider_fl_outs_corpus, spider_fl_outs_sents

    return spider_fl_outs
