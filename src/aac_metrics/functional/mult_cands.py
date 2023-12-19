#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Union

import torch
import tqdm

from torch import Tensor

from aac_metrics.utils.checks import is_mult_sents


SELECTIONS = ("max", "min", "mean")


def mult_cands_metric(
    metric: Callable,
    metric_out_name: str,
    mult_candidates: list[list[str]],
    mult_references: list[list[str]],
    return_all_scores: bool = True,
    return_all_cands_scores: bool = False,
    selection: str = "max",
    reduction: Callable[[Tensor], Tensor] = torch.mean,
    **kwargs,
) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
    """Multiple candidates metric wrapper.

    :param metric: Any Callable metric code. Take (candidates, mult_references, return_all_scores) and return the global and local scores.
    :param metric_out_name: The name of the metric output. Should be one of the keys of the sentences local scores returned by the metric.
    :param mult_candidates: The list of list of sentences to evaluate.
    :param mult_references: The references input.
    :param selection: The selection to apply. Can be "max", "min" or "mean". defaults to "max".
    :param reduction: The reduction function to apply to local scores. defaults to torch.mean.
    :param **kwargs: The keywords arguments given to the metric call.
    :returns: A tuple of globals and locals scores or a scalar tensor with the main global score.
    """
    if not is_mult_sents(mult_candidates):
        error_msg = f"Invalid mult_candidates type. (expected list[list[str]], found {mult_references.__class__.__name__})"
        raise ValueError(error_msg)

    if not is_mult_sents(mult_references):
        error_msg = f"Invalid mult_references type. (expected list[list[str]], found {mult_references.__class__.__name__})"
        raise ValueError(error_msg)

    if len(mult_candidates) <= 0:
        raise ValueError(
            f"Cannot compute max metric without at least 1 candidate. (found {len(mult_candidates)=})"
        )
    if len(mult_candidates) != len(mult_references):
        raise ValueError(
            f"Number of candidate and mult_references are different ({len(mult_candidates)} != {len(mult_references)})."
        )

    if selection not in SELECTIONS:
        raise ValueError(
            f"Invalid argument {selection=}. (expected one of {SELECTIONS})"
        )

    n_cands_per_audio = len(mult_candidates[0])
    if not all(len(cands) == n_cands_per_audio for cands in mult_candidates):
        raise ValueError(
            "Cannot compute multiple candidates metric with a various number of candidates."
        )

    all_sents_scores_lst: list[dict[str, Tensor]] = []
    verbose = kwargs.get("verbose", 0)

    for i in tqdm.trange(n_cands_per_audio, disable=verbose < 2):
        candidates_i = [cands[i] for cands in mult_candidates]
        _global_scores_i, sents_scores_i = metric(
            candidates_i,
            mult_references,
            return_all_scores=True,
            **kwargs,
        )
        all_sents_scores_lst.append(sents_scores_i)

    # list[dict[str, Tensor]] to dict[str, stacked Tensor]
    keys = list(all_sents_scores_lst[0].keys())
    all_sents_scores = {
        k: torch.stack([sents_scores_i[k] for sents_scores_i in all_sents_scores_lst])
        for k in keys
    }
    # all_sents_scores dict of tensor of shapes (n_cands_per_audio, n_items)

    if selection == "max":
        indexes = all_sents_scores[metric_out_name].argmax(dim=0).unsqueeze(dim=0)
        outs_sents = {
            f"{k}_{selection}": scores.gather(0, indexes).squeeze(dim=0)
            for k, scores in all_sents_scores.items()
        }

    elif selection == "min":
        indexes = all_sents_scores[metric_out_name].argmin(dim=0).unsqueeze(dim=0)
        outs_sents = {
            f"{k}_{selection}": scores.gather(0, indexes).squeeze(dim=0)
            for k, scores in all_sents_scores.items()
        }

    elif selection == "mean":
        selected_scores = all_sents_scores[metric_out_name].mean(dim=0)
        outs_sents = {f"{metric_out_name}_{selection}": selected_scores}

    else:
        raise ValueError(
            f"Invalid argument {selection=}. (expected one of {SELECTIONS})"
        )

    if return_all_cands_scores:
        outs_sents |= {
            f"{k}_all": scores.transpose(0, 1) for k, scores in all_sents_scores.items()
        }

    outs_corpus = {k: reduction(scores) for k, scores in outs_sents.items()}

    if return_all_scores:
        return outs_corpus, outs_sents
    else:
        out_key = f"{metric_out_name}_{selection}"
        return outs_corpus[out_key]
