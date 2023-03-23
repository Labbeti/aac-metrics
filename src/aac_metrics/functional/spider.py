#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Iterable, Optional, Union

from torch import Tensor

from aac_metrics.functional.cider_d import cider_d
from aac_metrics.functional.spice import spice


def spider(
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
    verbose: int = 0,
) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
    """SPIDEr function.

    - Paper: https://arxiv.org/pdf/1612.00370.pdf

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
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple of globals and locals scores or a scalar tensor with the main global score.
    """

    if len(candidates) != len(mult_references):
        raise ValueError(
            f"Number of candidates and mult_references are different (found {len(candidates)} != {len(mult_references)})."
        )

    cider_d_outs: tuple = cider_d(  # type: ignore
        candidates,
        mult_references,
        True,
        n=n,
        sigma=sigma,
        tokenizer=tokenizer,
        return_tfidf=return_tfidf,
    )
    spice_outs: tuple = spice(  # type: ignore
        candidates,
        mult_references,
        True,
        cache_path=cache_path,
        java_path=java_path,
        tmp_path=tmp_path,
        n_threads=n_threads,
        java_max_memory=java_max_memory,
        timeout=timeout,
        verbose=verbose,
    )
    spider_outs = _spider_from_outputs(cider_d_outs, spice_outs)

    if return_all_scores:
        return spider_outs
    else:
        return spider_outs[0]["spider"]


def _spider_from_outputs(
    cider_d_outs: tuple[dict[str, Tensor], dict[str, Tensor]],
    spice_outs: tuple[dict[str, Tensor], dict[str, Tensor]],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Combines CIDErD and SPICE outputs."""
    cider_d_outs_corpus, cider_d_outs_sents = cider_d_outs
    spice_outs_corpus, spice_outs_sents = spice_outs

    spider_score = (cider_d_outs_corpus["cider_d"] + spice_outs_corpus["spice"]) / 2.0
    spider_scores = (cider_d_outs_sents["cider_d"] + spice_outs_sents["spice"]) / 2.0

    spider_outs_corpus = (
        cider_d_outs_corpus | spice_outs_corpus | {"spider": spider_score}
    )
    spider_outs_sents = (
        cider_d_outs_sents
        | spice_outs_sents
        | {
            "spider": spider_scores,
        }
    )
    spider_outs = spider_outs_corpus, spider_outs_sents

    return spider_outs
