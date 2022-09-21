#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Union

import torch

from torch import Tensor


def diversity_ratio(
    candidates: list[str],
    mult_references: list[list[str]],
    return_all_scores: bool = True,
    tokenizer: Callable[[str], list[str]] = str.split,
) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Tensor]]]:
    tok_cands = list(map(tokenizer, candidates))
    tok_mrefs = [list(map(tokenizer, refs)) for refs in mult_references]

    cands_divs = torch.empty(len(candidates), dtype=torch.float64)
    mrefs_divs = torch.empty(len(candidates), dtype=torch.float64)

    for i, (cand, refs) in enumerate(zip(tok_cands, tok_mrefs)):
        cand_div = len(cand) / max(len(set(cand)), 1)
        refs_divs = [len(ref) / max(len(set(ref)), 1) for ref in refs]
        refs_div = sum(refs_divs) / max(len(refs_divs), 1)

        cands_divs[i] = cand_div
        mrefs_divs[i] = refs_div

    diversity_ratios = cands_divs / torch.max(mrefs_divs, torch.ones_like(mrefs_divs))

    cands_div = cands_divs.mean()
    mrefs_div = mrefs_divs.mean()
    diversity_ratio = diversity_ratios.mean()

    if return_all_scores:
        global_scores = {
            "diversity_ratio": diversity_ratio,
            "cands_diversity": cands_div,
            "mrefs_diversity": mrefs_div,
        }
        local_scores = {
            "diversity_ratio": diversity_ratios,
            "cands_diversity": cands_divs,
            "mrefs_diversity": mrefs_divs,
        }
        return global_scores, local_scores
    else:
        return diversity_ratio
