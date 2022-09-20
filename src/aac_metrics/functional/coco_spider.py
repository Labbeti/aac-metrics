#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Optional, Union

from torch import Tensor

from aac_metrics.functional.coco_cider_d import coco_cider_d
from aac_metrics.functional.coco_spice import coco_spice


def coco_spider(
    candidates: list[str],
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
    if len(candidates) != len(mult_references):
        raise ValueError(
            f"Number of hypothesis and mult_references are different ({len(candidates)} != {len(mult_references)})."
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
        assert isinstance(cider_d_out, tuple)
        assert isinstance(spice_out, tuple)
        cider_d_global_outs, cider_d_local_outs = cider_d_out
        spice_global_outs, spice_local_outs = spice_out
        spider_agg_outs = {
            "cider_d": cider_d_global_outs["cider_d"],
            "spice": spice_global_outs["spice"],
            "spider": (cider_d_global_outs["cider_d"] + spice_global_outs["spice"])
            / 2.0,
        }
        spider_loc_outs = {
            "cider_d": cider_d_local_outs["cider_d"],
            "spice": spice_local_outs["spice"],
            "spider": (cider_d_local_outs["cider_d"] + spice_local_outs["spice"]) / 2.0,
        }
        return spider_agg_outs, spider_loc_outs
    else:
        assert isinstance(cider_d_out, Tensor)
        assert isinstance(spice_out, Tensor)
        return (cider_d_out + spice_out) / 2.0
