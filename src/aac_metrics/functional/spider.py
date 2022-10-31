#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Optional, Union

from torch import Tensor

from aac_metrics.functional.coco_cider_d import coco_cider_d
from aac_metrics.functional.coco_spice import coco_spice


def spider(
    candidates: list[str],
    mult_references: list[list[str]],
    return_all_scores: bool = True,
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
    """SPIDEr function.

    Paper: https://arxiv.org/pdf/1612.00370.pdf

    :param candidates: The list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target.
    :param return_all_scores: If True, returns a tuple containing the globals and locals scores.
        Otherwise returns a scalar tensor containing the main global score.
        defaults to True.
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

    if len(candidates) != len(mult_references):
        raise ValueError(
            f"Number of candidates and mult_references are different (found {len(candidates)} != {len(mult_references)})."
        )

    cider_d_out = coco_cider_d(
        candidates,
        mult_references,
        return_all_scores,
        n=n,
        sigma=sigma,
        tokenizer=tokenizer,
        return_tfidf=return_tfidf,
    )
    spice_out = coco_spice(
        candidates,
        mult_references,
        return_all_scores,
        java_path=java_path,
        tmp_path=tmp_path,
        cache_path=cache_path,
        n_threads=n_threads,
        java_max_memory=java_max_memory,
        verbose=verbose,
    )

    if return_all_scores:
        assert isinstance(cider_d_out, tuple), "INTERNAL error type."
        assert isinstance(spice_out, tuple), "INTERNAL error type."
        cider_d_global_scores, cider_d_local_scores = cider_d_out
        spice_global_scores, spice_local_scores = spice_out

        spider_global_scores = {
            "cider_d": cider_d_global_scores["cider_d"],
            "spice": spice_global_scores["spice"],
            "spider": (cider_d_global_scores["cider_d"] + spice_global_scores["spice"])
            / 2.0,
        }
        spider_local_scores = {
            "cider_d": cider_d_local_scores["cider_d"],
            "spice": spice_local_scores["spice"],
            "spider": (cider_d_local_scores["cider_d"] + spice_local_scores["spice"])
            / 2.0,
        }
        return spider_global_scores, spider_local_scores
    else:
        assert isinstance(cider_d_out, Tensor), "INTERNAL error type."
        assert isinstance(spice_out, Tensor), "INTERNAL error type."
        return (cider_d_out + spice_out) / 2.0
