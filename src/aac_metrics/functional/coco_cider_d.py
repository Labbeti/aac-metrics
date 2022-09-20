#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

from collections import defaultdict, Counter
from typing import Any, Callable, Union

import numpy as np
import torch

from torch import Tensor


def coco_cider_d(
    candidates: list[str],
    mult_references: list[list[str]],
    return_all_scores: bool = False,
    n: int = 4,
    sigma: float = 6.0,
    tokenizer: Callable[[str], list[str]] = str.split,
    return_tfidf: bool = False,
) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Any]]]:
    """
    :param n: set cider to sum over 1 to 4-grams
    :param sigma: set the standard deviation parameter for gaussian penalty
    """
    if len(candidates) != len(mult_references):
        raise ValueError(
            f"Invalid number of candidates and references. (found {len(candidates)=} != {len(mult_references)=})"
        )

    if len(candidates) <= 1:
        raise ValueError(
            f"CIDEr metric does not support less than 2 candidates with 2 references. (found {len(candidates)=}, but expected > 1)"
        )

    cooked_mrefs = [
        _cook_references(refs, n=n, tokenizer=tokenizer) for refs in mult_references
    ]
    cooked_cands = [
        _cook_candidate(cand, n=n, tokenizer=tokenizer) for cand in candidates
    ]

    cider_d_score, cider_d_scores, tfidf_lst = _compute_score(
        cooked_cands,
        cooked_mrefs,
        n,
        sigma,
    )

    cider_d_score = torch.as_tensor(cider_d_score)
    cider_d_scores = torch.from_numpy(cider_d_scores)

    if return_all_scores:
        cider_d_global_outs = {
            "cider_d": cider_d_score,
        }
        cider_d_local_outs = {
            "cider_d": cider_d_scores,
        }
        if return_tfidf:
            cider_d_local_outs["tfidf_lst"] = tfidf_lst  # type: ignore

        return cider_d_global_outs, cider_d_local_outs
    else:
        return cider_d_score


def _precook(
    sentence: str,
    n: int,
    tokenizer: Callable[[str], list[str]] = str.split,
) -> Counter[tuple[str, ...]]:
    """
    Takes a string as input and returns an object that can be given to
    either cook_refs or cook_test. This is optional: cook_refs and cook_test
    can take string arguments as well.
    :param s: string : sentence to be converted into ngrams
    :param n: int    : number of ngrams for which representation is calculated
    :return: term frequency vector for occuring ngrams
    """
    words = tokenizer(sentence)
    counter = Counter()
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i : i + k])
            counter[ngram] += 1
    return counter


def _cook_references(
    references: list[str],
    n: int,
    tokenizer: Callable[[str], list[str]] = str.split,
) -> list[Counter[tuple[str, ...]]]:  # lhuang: oracle will call with 'average'
    """Takes a list of reference sentences for a single segment
    and returns an object that encapsulates everything that BLEU
    needs to know about them.
    :param refs: list of string : reference sentences for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (list of dict)
    """
    return [_precook(ref, n, tokenizer) for ref in references]


def _cook_candidate(
    candidate: str,
    n: int,
    tokenizer: Callable[[str], list[str]] = str.split,
) -> Counter[tuple[str, ...]]:
    """Takes a test sentence and returns an object that
    encapsulates everything that BLEU needs to know about it.
    :param test: list of string : hypothesis sentence for some image
    :param n: int : number of ngrams for which (ngram) representation is calculated
    :return: result (dict)
    """
    return _precook(candidate, n, tokenizer)


def _compute_doc_freq(cooked_mrefs: list[list[Counter]]) -> tuple[Counter, float]:
    """
    Compute term frequency for reference data.
    This will be used to compute idf (inverse document frequency later)
    The term frequency is stored in the object
    :return: None
    """
    document_frequency = Counter()
    for refs in cooked_mrefs:
        # refs, k ref captions of one image
        for ngram in set([ngram for ref in refs for ngram, count in ref.items()]):
            document_frequency[ngram] += 1
    # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)

    # compute log reference length
    log_ref_len = np.log(float(len(cooked_mrefs)))
    return document_frequency, log_ref_len


def _counter_to_vec(
    counters: dict[tuple, int], log_ref_len: float, n: int, document_frequency: Counter
) -> tuple[list[defaultdict], list, int]:
    """
    Function maps counts of ngram to vector of tfidf weights.
    The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
    The n-th entry of array denotes length of n-grams.
    :param cnts:
    :return: vec (array of dict), norm (array of float), length (int)
    """
    vec = [defaultdict(float) for _ in range(n)]
    length = 0
    norm = [0.0 for _ in range(n)]
    for (ngram, term_freq) in counters.items():
        # give word count 1 if it doesn't appear in reference corpus
        log_df = np.log(max(1.0, document_frequency[ngram]))

        # ngram index
        n = len(ngram) - 1

        # tf (term_freq) * idf (precomputed idf) for n-grams
        vec[n][ngram] = float(term_freq) * (log_ref_len - log_df)

        # compute norm for the vector.  the norm will be used for computing similarity
        norm[n] += pow(vec[n][ngram], 2)

        if n == 1:
            length += term_freq
    norm = [np.sqrt(n) for n in norm]
    return vec, norm, length


def _similarity(
    cand_vec: list[defaultdict],
    ref_vec: list[defaultdict],
    cand_norm: list[float],
    ref_norm: list[float],
    cand_len: int,
    ref_len: int,
    n: int,
    sigma: float,
) -> np.ndarray:
    """
    Compute the cosine similarity of two vectors.
    :param vec_hyp: array of dictionary for vector corresponding to hypothesis
    :param vec_ref: array of dictionary for vector corresponding to reference
    :param norm_hyp: array of float for vector corresponding to hypothesis
    :param norm_ref: array of float for vector corresponding to reference
    :param length_hyp: int containing length of hypothesis
    :param length_ref: int containing length of reference
    :return: array of score for each n-grams cosine similarity
    """
    delta = float(cand_len - ref_len)
    # measure consine similarity
    val = np.array([0.0 for _ in range(n)])
    for n in range(n):
        # ngram
        for (ngram, _count) in cand_vec[n].items():
            # vrama91 : added clipping
            val[n] += min(cand_vec[n][ngram], ref_vec[n][ngram]) * ref_vec[n][ngram]

        if (cand_norm[n] != 0) and (ref_norm[n] != 0):
            val[n] /= cand_norm[n] * ref_norm[n]

        assert not math.isnan(val[n])
        # vrama91: added a length based gaussian penalty
        val[n] *= np.e ** (-(delta**2) / (2 * sigma**2))
    return val


def _compute_cider(
    cooked_cands: list,
    cooked_mrefs: list,
    document_frequency: Counter,
    log_ref_len: float,
    n: int,
    sigma: float,
) -> tuple[list[float], list[tuple]]:
    scores = []
    tfidf_lst = []
    for test, refs in zip(cooked_cands, cooked_mrefs):
        # compute vector for test captions
        vec, norm, length = _counter_to_vec(test, log_ref_len, n, document_frequency)
        # compute vector for ref captions
        score = np.array([0.0 for _ in range(n)])
        vec_refs = []
        for ref in refs:
            vec_ref, norm_ref, length_ref = _counter_to_vec(
                ref, log_ref_len, n, document_frequency
            )
            vec_refs.append(vec_ref)
            score += _similarity(
                vec, vec_ref, norm, norm_ref, length, length_ref, n, sigma
            )
        # change by vrama91 - mean of ngram scores, instead of sum
        score_avg = np.mean(score)
        # divide by number of mult_references
        score_avg /= len(refs)
        # multiply score by 10
        score_avg *= 10.0
        # append score of an image to the score list
        scores.append(score_avg)
        tfidf_lst.append((vec, vec_refs))
    return scores, tfidf_lst


def _compute_score(
    cooked_cands: list[Counter],
    cooked_mrefs: list[list[Counter]],
    n: int,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, list[tuple]]:
    # compute idf
    document_frequency, log_ref_len = _compute_doc_freq(cooked_mrefs)
    # sanity check: assert to check document frequency
    assert len(cooked_cands) >= max(document_frequency.values())
    # compute cider score
    scores, tfidf_lst = _compute_cider(
        cooked_cands,
        cooked_mrefs,
        document_frequency,
        log_ref_len,
        n,
        sigma,
    )
    scores = np.array(scores)
    score = np.mean(scores)
    return score, scores, tfidf_lst
