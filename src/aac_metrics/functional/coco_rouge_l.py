#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Description : Computes ROUGE-L metric as described by Lin and Hovey (2004)
#
# Creation Date : 2015-01-07 06:03
# Author : Ramakrishna Vedantam <vrama91@vt.edu>

# =================================================================
# This code was pulled from https://github.com/tylin/coco-caption
# Image-specific names and comments have been changed to be audio-specific
# =================================================================

import logging

from typing import Callable, Union

import numpy as np
import torch

from torch import Tensor


logger = logging.getLogger(__name__)


def coco_rouge_l(
    candidates: list[str],
    mult_references: list[list[str]],
    return_all_scores: bool = True,
    beta: float = 1.2,
    tokenizer: Callable[[str], list[str]] = str.split,
) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
    """Recall-Oriented Understudy for Gisting Evaluation function.

    Paper: https://aclanthology.org/W04-1013.pdf

    :param candidates: The list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target.
    :param return_all_scores: If True, returns a tuple containing the globals and locals scores.
        Otherwise returns a scalar tensor containing the main global score.
        defaults to True.
    :param beta: Determines the weight of recall in the combined f-score. defaults to 1.2.
    :param tokenizer: The fast tokenizer used to split sentences into words. defaults to str.split.
    :returns: A tuple of globals and locals scores or a scalar tensor with the main global score.
    """
    rouge_l_scores = _coco_rouge_l_update(
        candidates, mult_references, beta, tokenizer, []
    )
    return _coco_rouge_l_compute(rouge_l_scores, return_all_scores)


def _coco_rouge_l_update(
    candidates: list[str],
    mult_references: list[list[str]],
    beta: float,
    tokenizer: Callable[[str], list[str]],
    prev_rouge_l_scores: list[float],
) -> list[float]:
    if len(candidates) != len(mult_references):
        raise ValueError(
            f"Invalid number of candidates and references. (found {len(candidates)=} != {len(mult_references)=})"
        )
    new_rouge_l_scores = [
        __calc_score(cand, refs, beta, tokenizer)
        for cand, refs in zip(candidates, mult_references)
    ]
    prev_rouge_l_scores += new_rouge_l_scores
    return prev_rouge_l_scores


def _coco_rouge_l_compute(
    rouge_l_scores: list[float],
    return_all_scores: bool,
) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
    # Note: use numpy to compute mean because np.mean and torch.mean can give very small differences
    rouge_l_scores_np = np.array(rouge_l_scores)
    rouge_l_score_np = rouge_l_scores_np.mean()

    rouge_l_score_pt = torch.as_tensor(rouge_l_score_np)
    rouge_l_scores_pt = torch.from_numpy(rouge_l_scores_np)

    if return_all_scores:
        global_scores = {
            "rouge_l": rouge_l_score_pt,
        }
        local_scores = {
            "rouge_l": rouge_l_scores_pt,
        }
        return global_scores, local_scores
    else:
        return rouge_l_score_pt


def __calc_score(
    candidate: str,
    references: list[str],
    beta: float,
    tokenizer: Callable[[str], list[str]] = str.split,
) -> float:
    """Compute ROUGE-L score given one candidate and mult_references for an audio
    :param candidate: list of str : candidate sentence to be evaluated
    :param refs: list of str : COCO reference sentences for the particular audio to be evaluated
    :returns score: int (ROUGE-L score for the candidate evaluated against mult_references)
    """
    assert len(references) > 0
    prec = []
    rec = []

    # split into tokens
    token_c = tokenizer(candidate)

    # Add Labbeti: returns 0 when candidate is empty.
    if len(token_c) == 0:
        return 0.0

    for reference in references:
        # split into tokens
        token_r = tokenizer(reference)
        # compute the longest common subsequence
        lcs = __my_lcs(token_r, token_c)
        prec.append(lcs / float(len(token_c)))
        rec.append(lcs / float(len(token_r)))

    prec_max = max(prec)
    rec_max = max(rec)

    if prec_max != 0 and rec_max != 0:
        score = ((1 + beta**2) * prec_max * rec_max) / float(
            rec_max + beta**2 * prec_max
        )
    else:
        score = 0.0
    return score


def __my_lcs(string: list[str], sub: list[str]) -> int:
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings
    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if len(string) < len(sub):
        sub, string = string, sub

    lengths = [[0 for i in range(0, len(sub) + 1)] for j in range(0, len(string) + 1)]

    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if string[i - 1] == sub[j - 1]:
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

    return lengths[len(string)][len(sub)]
