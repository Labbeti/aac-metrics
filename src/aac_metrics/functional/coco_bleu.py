#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

from collections import Counter
from typing import Any, Callable, Optional, Union

import torch

from torch import Tensor


BLEU_COCO_OPTIONS = ("shortest", "average", "closest")


def coco_bleu(
    candidates: list[str],
    mult_references: list[list[str]],
    return_all_scores: bool = True,
    n: int = 4,
    option: str = "closest",
    verbose: int = 0,
    tokenizer: Callable[[str], list[str]] = str.split,
    return_1_to_n: bool = False,
) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
    """BiLingual Evaluation Understudy function.

    Paper: https://www.aclweb.org/anthology/P02-1040.pdf

    :param candidates: The list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target.
    :param return_all_scores: If True, returns a tuple containing the globals and locals scores.
        Otherwise returns a scalar tensor containing the main global score.
        defaults to True.
    :param n: Maximal number of n-grams taken into account. defaults to 4.
    :param option: Corpus reference length mode. Can be "shortest", "average" or "closest". defaults to "closest".
    :param verbose: The verbose level. defaults to 0.
    :param tokenizer: The fast tokenizer used to split sentences into words. defaults to str.split.
    :param return_1_to_no: If True, returns the n-grams results from 1 to n.
        Otherwise return the n-grams scores.
        defauts to False.
    :returns: A tuple of globals and locals scores or a scalar tensor with the main global score.
    """
    cooked_cands, cooked_mrefs = _coco_bleu_update(
        candidates,
        mult_references,
        n,
        tokenizer,
        [],
        [],
    )
    return _coco_bleu_compute(
        cooked_cands,
        cooked_mrefs,
        return_all_scores,
        n,
        option,
        verbose,
        return_1_to_n,
    )


def _coco_bleu_update(
    candidates: list[str],
    mult_references: list[list[str]],
    n: int,
    tokenizer: Callable[[str], list[str]],
    prev_cooked_cands: list,
    prev_cooked_mrefs: list,
) -> tuple[list, list[tuple]]:
    if len(candidates) != len(mult_references):
        raise ValueError(
            f"Invalid number of candidates and references. (found {len(candidates)=} != {len(mult_references)=})"
        )
    new_cooked_mrefs = [
        __cook_references(refs, None, n, tokenizer) for refs in mult_references
    ]
    new_cooked_cands = [
        __cook_candidate(cand, cooked_mrefs_i, None, n, tokenizer)
        for cand, cooked_mrefs_i in zip(candidates, new_cooked_mrefs)
    ]
    prev_cooked_cands += new_cooked_cands
    prev_cooked_mrefs += new_cooked_mrefs
    return prev_cooked_cands, prev_cooked_mrefs


def _coco_bleu_compute(
    cooked_cands: list,
    cooked_mrefs: list,
    return_all_scores: bool = True,
    n: int = 4,
    option: str = "closest",
    verbose: int = 0,
    return_1_to_n: bool = False,
) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Tensor]]]:
    if option not in BLEU_COCO_OPTIONS:
        raise ValueError(
            f"Invalid option {option=}. (expected one of {BLEU_COCO_OPTIONS})"
        )

    score_1_to_n, scores_1_to_n = __compute_bleu_score(
        cooked_cands,
        cooked_mrefs,
        n,
        option=option,
        verbose=verbose,
    )

    # Note: we use f64 because the original implem use numpy which uses f64 precision
    dtype = torch.float64
    score_n = torch.as_tensor(score_1_to_n[-1], dtype=dtype)
    scores_n = torch.as_tensor(scores_1_to_n[-1], dtype=dtype)
    score_1_to_n = torch.as_tensor(score_1_to_n, dtype=dtype)
    scores_1_to_n = torch.as_tensor(scores_1_to_n, dtype=dtype)

    if return_all_scores:
        global_scores = {
            f"bleu_{n}": score_n,
        }
        local_scores = {
            f"bleu_{n}": scores_n,
        }

        if return_1_to_n:
            global_scores[f"bleu_1_to_{n}"] = score_1_to_n
            local_scores[f"bleu_1_to_{n}"] = scores_1_to_n

        return global_scores, local_scores
    else:
        return score_n


def __cook_sentence(
    s: str,
    n: int = 4,
    tokenizer: Callable[[str], list[str]] = str.split,
) -> tuple[int, Counter[tuple[str, ...]]]:
    """Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well."""
    words = tokenizer(s)
    counts = Counter()
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i : i + k])
            counts[ngram] += 1
    return len(words), counts


def __cook_references(
    refs: list[str],
    eff: Optional[str],
    n: int,
    tokenizer: Callable[[str], list[str]],
) -> tuple[Union[float, list], dict]:  # lhuang: oracle will call with "average"
    """Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them."""

    reflen = []
    maxcounts = {}
    for ref in refs:
        rl, counts = __cook_sentence(ref, n, tokenizer)
        reflen.append(rl)
        for (ngram, count) in counts.items():
            maxcounts[ngram] = max(maxcounts.get(ngram, 0), count)

    # Calculate effective reference sentence length.
    if eff == "shortest":
        reflen = min(reflen)
    elif eff == "average":
        reflen = float(sum(reflen)) / len(reflen)

    # lhuang: N.B.: leave reflen computaiton to the very end!!
    # lhuang: N.B.: in case of "closest", keep a list of reflens!! (bad design)
    return (reflen, maxcounts)


def __cook_candidate(
    test: str,
    reflen_refmaxcounts: tuple[Any, dict[tuple[str, ...], int]],
    eff: Optional[None],
    n: int,
    tokenizer: Callable[[str], list[str]],
) -> dict[str, Any]:
    """Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it."""

    testlen, counts = __cook_sentence(test, n, tokenizer)
    reflen, refmaxcounts = reflen_refmaxcounts  # Replaces the tuple unpacking

    result = {}
    # Calculate effective reference sentence length.
    if eff == "closest":
        result["reflen"] = min((abs(len - testlen), len) for len in reflen)[1]
    else:  # i.e., "average" or "shortest" or None
        result["reflen"] = reflen

    result["testlen"] = testlen
    result["guess"] = [max(0, testlen - k + 1) for k in range(1, n + 1)]
    result["correct"] = [0] * n

    for (ngram, count) in counts.items():
        result["correct"][len(ngram) - 1] += min(refmaxcounts.get(ngram, 0), count)

    return result


def __compute_bleu_score(
    cooked_cands: list,
    cooked_mrefs: list,
    n: int,
    option: Optional[str] = "closest",
    verbose: int = 0,
) -> tuple[list[float], list[list[float]]]:
    SMALL = 1e-9
    TINY = 1e-15  # so that if guess is 0 still return 0
    bleu_list = [[] for _ in range(n)]

    if option is None:
        option = "average" if len(cooked_mrefs) == 1 else "closest"

    global_cands_len = 0
    global_mrefs_len = 0
    totalcomps = {"testlen": 0, "reflen": 0, "guess": [0] * n, "correct": [0] * n}

    # for each sentence
    for comps in cooked_cands:
        testlen = comps["testlen"]
        global_cands_len += testlen

        reflen = __single_reflen(comps["reflen"], option, testlen)

        global_mrefs_len += reflen

        for key in ("guess", "correct"):
            for k in range(n):
                totalcomps[key][k] += comps[key][k]

        # append per audio bleu score
        bleu = 1.0
        for k in range(n):
            bleu *= (float(comps["correct"][k]) + TINY) / (
                float(comps["guess"][k]) + SMALL
            )
            bleu_list[k].append(bleu ** (1.0 / (k + 1)))

        # N.B.: avoid zero division
        ratio = (testlen + TINY) / (reflen + SMALL)
        if ratio < 1:
            for k in range(n):
                bleu_list[k][-1] *= math.exp(1 - 1 / ratio)

        if verbose > 1:
            print(comps, reflen)

    totalcomps["reflen"] = global_mrefs_len
    totalcomps["testlen"] = global_cands_len

    bleus = []
    bleu = 1.0
    for k in range(n):
        bleu *= float(totalcomps["correct"][k] + TINY) / (
            totalcomps["guess"][k] + SMALL
        )
        bleus.append(bleu ** (1.0 / (k + 1)))
    ratio = (global_cands_len + TINY) / (
        global_mrefs_len + SMALL
    )  # N.B.: avoid zero division
    if ratio < 1:
        for k in range(n):
            bleus[k] *= math.exp(1 - 1 / ratio)

    if verbose > 0:
        print(totalcomps)
        print("ratio:", ratio)

    return bleus, bleu_list


def __single_reflen(
    reflens: list[int],
    option: Optional[str] = None,
    testlen: Optional[int] = None,
) -> float:
    if option == "shortest":
        reflen = min(reflens)
    elif option == "average":
        reflen = float(sum(reflens)) / len(reflens)
    elif option == "closest":
        assert testlen is not None
        reflen = min((abs(len - testlen), len) for len in reflens)[1]
    else:
        raise ValueError(
            f"Invalid argument {option=}. (expected one of {BLEU_COCO_OPTIONS})"
        )

    return reflen
