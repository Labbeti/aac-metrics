#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BASED ON https://github.com/blmoistawinde/fense/
"""

import logging

from typing import Callable, Iterable, Optional, Union

import torch

from torch import Tensor
from transformers.models.auto.tokenization_auto import AutoTokenizer

from aac_metrics.functional.fluerr import (
    fluerr,
    _load_echecker_and_tokenizer,
    BERTFlatClassifier,
)
from aac_metrics.functional.spider import spider


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
    cache_path: str = "$HOME/.cache",
    java_path: str = "java",
    tmp_path: str = "/tmp",
    n_threads: Optional[int] = None,
    java_max_memory: str = "8G",
    timeout: Union[None, int, Iterable[int]] = None,
    # FluencyError args
    echecker: Union[str, BERTFlatClassifier] = "echecker_clotho_audiocaps_base",
    echecker_tokenizer: Optional[AutoTokenizer] = None,
    error_threshold: float = 0.9,
    device: Union[str, torch.device, None] = "auto",
    batch_size: int = 32,
    reset_state: bool = True,
    return_probs: bool = True,
    # Other args
    penalty: float = 0.9,
    verbose: int = 0,
) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Tensor]]]:
    """Combinaison of SPIDEr with Fluency Error detector.

    Based on https://github.com/felixgontier/dcase-2023-baseline/blob/main/metrics.py#L48.

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
    :param cache_path: The path to the external code directory. defaults to "$HOME/.cache".
    :param java_path: The path to the java executable. defaults to "java".
    :param tmp_path: Temporary directory path. defaults to "/tmp".
    :param java_max_memory: The maximal java memory used. defaults to "8G".
    :param n_threads: Number of threads used to compute SPICE.
        None value will use the default value of the java program.
        defaults to None.
    :param echecker: The echecker model used to detect fluency errors.
        Can be "echecker_clotho_audiocaps_base", "echecker_clotho_audiocaps_tiny", "none" or None.
        defaults to "echecker_clotho_audiocaps_base".
    :param echecker_tokenizer: The tokenizer of the echecker model.
        If None and echecker is not None, this value will be inferred with `echecker.model_type`.
        defaults to None.
    :param error_threshold: The threshold used to detect fluency errors for echecker model. defaults to 0.9.
    :param device: The PyTorch device used to run FENSE models. If "auto", it will use cuda if available. defaults to "auto".
    :param batch_size: The batch size of the sBERT and echecker models. defaults to 32.
    :param reset_state: If True, reset the state of the PyTorch global generator after the pre-trained model are built. defaults to True.
    :param return_probs: If True, return each individual error probability given by the fluency detector model. defaults to True.
    :param penalty: The penalty coefficient applied. Higher value means to lower the cos-sim scores when an error is detected. defaults to 0.9.
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple of globals and locals scores or a scalar tensor with the main global score.
    """

    # Init models
    echecker, echecker_tokenizer = _load_echecker_and_tokenizer(
        echecker, echecker_tokenizer, device, reset_state, verbose
    )

    spider_outs: tuple = spider(candidates, mult_references, True, n, sigma, tokenizer, return_tfidf, cache_path, java_path, tmp_path, n_threads, java_max_memory, timeout, verbose)  # type: ignore
    fluerr_outs: tuple = fluerr(candidates, True, echecker, echecker_tokenizer, error_threshold, device, batch_size, reset_state, return_probs, verbose)  # type: ignore
    spider_fl_outs = _spider_fl_from_outputs(spider_outs, fluerr_outs, penalty)

    if return_all_scores:
        return spider_fl_outs
    else:
        return spider_fl_outs[0]["spider_fl"]


def _spider_fl_from_outputs(
    spider_outs: tuple[dict[str, Tensor], dict[str, Tensor]],
    fluerr_outs: tuple[dict[str, Tensor], dict[str, Tensor]],
    penalty: float = 0.9,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Combines SPIDEr and FluErr outputs.

    Based on https://github.com/felixgontier/dcase-2023-baseline/blob/main/metrics.py#L48
    """
    spider_outs_corpus, spider_outs_sents = spider_outs
    fluerr_outs_corpus, fluerr_outs_sents = fluerr_outs

    spider_scores = spider_outs_sents["spider"]
    fluerr_scores = fluerr_outs_sents["fluerr"]
    spider_fl_scores = spider_scores * (1.0 - penalty * fluerr_scores)
    spider_fl_score = spider_fl_scores.mean()

    spider_fl_outs_corpus = (
        spider_outs_corpus | fluerr_outs_corpus | {"spider_fl": spider_fl_score}
    )
    spider_fl_outs_sents = (
        spider_outs_sents | fluerr_outs_sents | {"spider_fl": spider_fl_scores}
    )
    spider_fl_outs = spider_fl_outs_corpus, spider_fl_outs_sents

    return spider_fl_outs
