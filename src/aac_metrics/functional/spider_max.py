#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Optional, Union

import torch

from torch import Tensor

from aac_metrics.functional.mult_cands import mult_cands_metric
from aac_metrics.functional.spider import spider


def spider_max(
    mult_candidates: list[list[str]],
    mult_references: list[list[str]],
    return_all_scores: bool = True,
    return_all_cands_scores: bool = False,
    # CIDEr args
    n: int = 4,
    sigma: float = 6.0,
    tokenizer: Callable[[str], list[str]] = str.split,
    return_tfidf: bool = False,
    # SPICE args
    cache_path: str = "$HOME/aac-metrics-cache",
    java_path: str = "java",
    tmp_path: str = "/tmp",
    n_threads: Optional[int] = None,
    java_max_memory: str = "8G",
    verbose: int = 0,
) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
    """SPIDEr-max function.

    Paper: https://hal.archives-ouvertes.fr/hal-03810396/file/Labbe_DCASE2022.pdf

    Compute the maximal SPIDEr score accross multiple candidates.

    :param mult_candidates: The list of list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target.
    :param return_all_scores: If True, returns a tuple containing the globals and locals scores.
        Otherwise returns a scalar tensor containing the main global score.
        defaults to True.
    :param return_all_cands_scores: If True, returns all multiple candidates scores in local_scores outputs as tensor of shape (n_audoi, n_cands_per_audio).
        defaults to False.
    :param n: Maximal number of n-grams taken into account. defaults to 4.
    :param sigma: Standard deviation parameter used for gaussian penalty. defaults to 6.0.
    :param tokenizer: The fast tokenizer used to split sentences into words. defaults to str.split.
    :param return_tfidf: If True, returns the list of dictionaries containing the tf-idf scores of n-grams in the local_score output.
        defaults to False.
    :param cache_path: The path to the external code directory. defaults to "$HOME/aac-metrics-cache".
    :param java_path: The path to the java executable. defaults to "java".
    :param tmp_path: Temporary directory path. defaults to "/tmp".
    :param java_max_memory: The maximal java memory used. defaults to "8G".
    :param n_threads: Number of threads used to compute SPICE.
        None value will use the default value of the java program.
        defaults to None.
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple of globals and locals scores or a scalar tensor with the main global score.
    """
    return mult_cands_metric(
        spider,
        "spider",
        mult_candidates,
        mult_references,
        return_all_scores,
        return_all_cands_scores,
        "max",
        torch.mean,
        # CIDEr args
        n=n,
        sigma=sigma,
        tokenizer=tokenizer,
        return_tfidf=return_tfidf,
        # SPICE args
        cache_path=cache_path,
        java_path=java_path,
        tmp_path=tmp_path,
        n_threads=n_threads,
        java_max_memory=java_max_memory,
        verbose=verbose,
    )
