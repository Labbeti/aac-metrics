#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Optional, Union

from torch import Tensor

from aac_metrics.functional.mult_cands import mult_cands_metric
from aac_metrics.functional.spider import spider


def spider_max(
    mult_candidates: list[list[str]],
    mult_references: list[list[str]],
    return_all_scores: bool = True,
    # CIDEr args
    n: int = 4,
    sigma: float = 6.0,
    tokenizer: Callable[[str], list[str]] = str.split,
    return_tfidf: bool = False,
    # SPICE args
    java_path: str = "java",
    tmp_path: str = "/tmp",
    cache_path: str = "$HOME/aac-metrics-cache",
    n_threads: Optional[int] = None,
    java_max_memory: str = "8G",
    verbose: int = 0,
) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Tensor]]]:
    return mult_cands_metric(
        spider,
        "spider",
        mult_candidates,
        mult_references,
        return_all_scores,
        "max",
        # CIDEr args
        n=n,
        sigma=sigma,
        tokenizer=tokenizer,
        return_tfidf=return_tfidf,
        # SPICE args
        java_path=java_path,
        tmp_path=tmp_path,
        cache_path=cache_path,
        n_threads=n_threads,
        java_max_memory=java_max_memory,
        verbose=verbose,
    )
