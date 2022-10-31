#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict, Counter
from typing import Any, Callable, Union

import numpy as np
import torch

from torch import Tensor


def coco_cider_d(
    candidates: list[str],
    mult_references: list[list[str]],
    return_all_scores: bool = True,
    n: int = 4,
    sigma: float = 6.0,
    tokenizer: Callable[[str], list[str]] = str.split,
    return_tfidf: bool = False,
) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Any]]]:
    """Consensus-based Image Description Evaluation function.

    Paper: https://arxiv.org/pdf/1411.5726.pdf

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
    :returns: A tuple of globals and locals scores or a scalar tensor with the main global score.
    """
    cooked_cands, cooked_mrefs = _coco_cider_d_update(
        candidates,
        mult_references,
        n,
        tokenizer,
        [],
        [],
    )
    return _coco_cider_d_compute(
        cooked_cands,
        cooked_mrefs,
        return_all_scores,
        n,
        sigma,
        return_tfidf,
    )


def _coco_cider_d_update(
    candidates: list[str],
    mult_references: list[list[str]],
    n: int,
    tokenizer: Callable[[str], list[str]],
    prev_cooked_cands: list,
    prev_cooked_mrefs: list,
) -> tuple[list, list]:
    if len(candidates) != len(mult_references):
        raise ValueError(
            f"Invalid number of candidates and references. (found {len(candidates)=} != {len(mult_references)=})"
        )
    new_cooked_mrefs = [
        [__cook_sentence(ref, n, tokenizer) for ref in refs] for refs in mult_references
    ]
    new_cooked_cands = [__cook_sentence(cand, n, tokenizer) for cand in candidates]
    prev_cooked_cands += new_cooked_cands
    prev_cooked_mrefs += new_cooked_mrefs
    return prev_cooked_cands, prev_cooked_mrefs


def _coco_cider_d_compute(
    cooked_cands: list,
    cooked_mrefs: list,
    return_all_scores: bool,
    n: int,
    sigma: float,
    return_tfidf: bool,
) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Any]]]:
    if len(cooked_cands) <= 1:
        raise ValueError(
            f"CIDEr metric does not support less than 2 candidates with 2 references. (found {len(cooked_cands)} candidates, but expected > 1)"
        )
    # compute idf
    document_frequency = __compute_doc_freq(cooked_mrefs)
    # compute log reference length
    log_ref_len = np.log(float(len(cooked_mrefs)))
    # sanity check: assert to check document frequency
    assert len(cooked_cands) >= max(document_frequency.values())
    # compute cider score
    cider_d_scores, tfidf_lst = __compute_cider(
        cooked_cands,
        cooked_mrefs,
        document_frequency,
        log_ref_len,
        n,
        sigma,
    )
    cider_d_score = cider_d_scores.mean()

    cider_d_scores = torch.from_numpy(cider_d_scores)
    cider_d_score = torch.as_tensor(cider_d_score, dtype=torch.float64)

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


def __cook_sentence(
    sentence: str,
    n: int,
    tokenizer: Callable[[str], list[str]],
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


def __compute_doc_freq(cooked_mrefs: list[list[Counter]]) -> Counter[tuple[str, ...]]:
    """
    Compute term frequency for reference data.
    This will be used to compute idf (inverse document frequency later)
    The term frequency is stored in the object
    :return: None
    """
    document_frequency = Counter()
    for refs in cooked_mrefs:
        all_refs_ngrams = set(ngram for ref in refs for ngram in ref.keys())
        for ngram in all_refs_ngrams:
            document_frequency[ngram] += 1

    return document_frequency


def __counter_to_vec(
    counters: dict[tuple, int],
    log_ref_len: float,
    n: int,
    document_frequency: Counter[tuple],
) -> tuple[list[defaultdict], np.ndarray, int]:
    """
    Function maps counts of ngram to vector of tfidf weights.
    The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
    The n-th entry of array denotes length of n-grams.
    :param cnts:
    :return: vec (array of dict), norm (array of float), length (int)
    """
    vec = [defaultdict(float) for _ in range(n)]
    length = 0
    norm = np.zeros((n,))

    for (ngram, term_freq) in counters.items():
        # give word count 1 if it doesn't appear in reference corpus
        log_df = np.log(max(1.0, document_frequency[ngram]))

        # ngram index
        cur_n = len(ngram) - 1

        # tf (term_freq) * idf (precomputed idf) for n-grams
        vec[cur_n][ngram] = float(term_freq) * (log_ref_len - log_df)

        # compute norm for the vector.  the norm will be used for computing similarity
        norm[cur_n] += pow(vec[cur_n][ngram], 2)

        if cur_n == 1:
            length += term_freq

    norm = np.sqrt(norm)
    return vec, norm, length


def __similarity(
    cand_vec: list[defaultdict],
    ref_vec: list[defaultdict],
    cand_norm: np.ndarray,
    ref_norm: np.ndarray,
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
    val = np.zeros((n,))

    for ni in range(n):
        # ngram
        for (ngram, count) in cand_vec[ni].items():
            # vrama91 : added clipping
            val[ni] += min(count, ref_vec[ni][ngram]) * ref_vec[ni][ngram]

        if (cand_norm[ni] != 0) and (ref_norm[ni] != 0):
            val[ni] /= cand_norm[ni] * ref_norm[ni]

        # vrama91: added a length based gaussian penalty
        val[ni] *= np.e ** (-(delta**2) / (2 * sigma**2))

    return val


def __compute_cider(
    cooked_cands: list,
    cooked_mrefs: list,
    document_frequency: Counter,
    log_ref_len: float,
    n: int,
    sigma: float,
    scale: float = 10.0,
) -> tuple[np.ndarray, list[tuple[list, list]]]:

    scores = np.empty((len(cooked_cands),))
    tfidf_lst = []

    for i, (test, refs) in enumerate(zip(cooked_cands, cooked_mrefs)):
        # compute vector for test captions
        vec, norm, length = __counter_to_vec(test, log_ref_len, n, document_frequency)
        # compute vector for ref captions
        ngrams_scores = np.zeros((n,))
        vec_refs = []
        for ref in refs:
            vec_ref, norm_ref, length_ref = __counter_to_vec(
                ref, log_ref_len, n, document_frequency
            )
            vec_refs.append(vec_ref)
            ngrams_scores += __similarity(
                vec, vec_ref, norm, norm_ref, length, length_ref, n, sigma
            )
        # change by vrama91 - mean of ngram scores, instead of sum
        score_avg = np.mean(ngrams_scores)
        # divide by number of mult_references
        score_avg /= len(refs)
        # multiply score by 10
        score_avg *= scale
        # append score of an image to the score list
        scores[i] = score_avg
        tfidf_lst.append((vec, vec_refs))

    return scores, tfidf_lst
