#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Union

import torch

from torch import Tensor


def mult_cands_wrapper(
    metric: Callable,
    mult_candidates: list[list[str]],
    mult_references: list[list[str]],
    reduction: Union[str, Callable[[Tensor], Tensor]] = "max",
    *args,
    **kwargs,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Multiple candidates metric wrapper.

    :param metric: Any Callable metric code.
    :param mult_candidates: The candidates input. Currently only supports having the same number of multiple candidates.
    :param mult_references: The references input.
    :param reduction: The reduction function to apply. Take Tensor of shape (n_cand_per_audio, n_audio) and reduce to (n_audio,).
        defaults to "max".
    :param metric_prefix: The prefix added to the output metric names. defaults to "mcands.".
    :param metric_suffix: The suffix added to the output metric names. defaults to ".max".
    :param *args: The positional arguments given to the metric call.
    :param **kwargs: The keywords arguments given to the metric call.
    """
    if len(mult_candidates) <= 0:
        raise ValueError(
            f"Cannot compute max metric without at least 1 hypothesis. (found {len(mult_candidates)=})"
        )
    if len(mult_candidates) != len(mult_references):
        raise ValueError(
            f"Number of hypothesis and mult_references are different ({len(mult_candidates)} != {len(mult_references)})."
        )

    n_cands_per_audio = len(mult_candidates[0])
    if not all(len(cands) == n_cands_per_audio for cands in mult_candidates):
        raise ValueError(
            "Cannot compute multiple candidates metric with a various number of candidates."
        )

    if isinstance(reduction, str):
        if reduction == "max":
            reduction = _max_reduce
        else:
            raise ValueError(f"Invalid argument {reduction=}.")

    all_local_scores_lst: list[dict[str, Tensor]] = []

    for i in range(n_cands_per_audio):
        candidates_i = [cands[i] for cands in mult_candidates]
        _global_scores_i, local_scores_i = metric(
            candidates_i, mult_references, *args, return_all_scores=True, **kwargs
        )
        all_local_scores_lst.append(local_scores_i)

    keys = list(all_local_scores_lst[0].keys())
    all_local_scores = {
        k: torch.stack([local_scores_i[k] for local_scores_i in all_local_scores_lst])
        for k in keys
    }
    local_scores = {k: reduction(scores) for k, scores in all_local_scores.items()}
    global_scores = {k: scores.mean() for k, scores in local_scores.items()}
    return local_scores, global_scores


def _max_reduce(x: Tensor) -> Tensor:
    return x.max(dim=0).values
