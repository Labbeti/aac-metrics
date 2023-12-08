#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Callable, Iterable, Optional, Union

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
    cache_path: Union[str, Path, None] = None,
    java_path: Union[str, Path, None] = None,
    tmp_path: Union[str, Path, None] = None,
    n_threads: Optional[int] = None,
    java_max_memory: str = "8G",
    timeout: Union[None, int, Iterable[int]] = None,
    verbose: int = 0,
) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
    """SPIDEr-max function.

    Compute the maximal SPIDEr score accross multiple candidates.

    - Paper: https://dcase.community/documents/workshop2022/proceedings/DCASE2022Workshop_Labbe_46.pdf

    .. warning::
        This metric requires at least 2 candidates with 2 sets of references, otherwise it will raises a ValueError.

    :param mult_candidates: The list of list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target.
    :param return_all_scores: If True, returns a tuple containing the globals and locals scores.
        Otherwise returns a scalar tensor containing the main global score.
        defaults to True.
    :param return_all_cands_scores: If True, returns all multiple candidates scores in sents_scores outputs as tensor of shape (n_audio, n_cands_per_audio).
        defaults to False.
    :param n: Maximal number of n-grams taken into account. defaults to 4.
    :param sigma: Standard deviation parameter used for gaussian penalty. defaults to 6.0.
    :param tokenizer: The fast tokenizer used to split sentences into words. defaults to str.split.
    :param return_tfidf: If True, returns the list of dictionaries containing the tf-idf scores of n-grams in the sents_score output.
        defaults to False.
    :param cache_path: The path to the external code directory. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_cache_path`.
    :param java_path: The path to the java executable. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_java_path`.
    :param tmp_path: Temporary directory path. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_tmp_path`.
    :param java_max_memory: The maximal java memory used. defaults to "8G".
    :param n_threads: Number of threads used to compute SPICE.
        None value will use the default value of the java program.
        defaults to None.
    :param timeout: The number of seconds before killing the java subprogram.
        If a list is given, it will restart the program if the i-th timeout is reached.
        If None, no timeout will be used.
        defaults to None.
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple of globals and locals scores or a scalar tensor with the main global score.
    """
    return mult_cands_metric(
        metric=spider,
        metric_out_name="spider",
        mult_candidates=mult_candidates,
        mult_references=mult_references,
        return_all_scores=return_all_scores,
        return_all_cands_scores=return_all_cands_scores,
        selection="max",
        reduction=torch.mean,
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
        timeout=timeout,
        verbose=verbose,
    )
