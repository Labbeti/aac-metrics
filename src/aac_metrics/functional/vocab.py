#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Callable, Literal, TypedDict, TypeVar, Union

import torch
from torch import Generator, Tensor

from aac_metrics.utils.checks import check_metric_inputs, is_mono_sents

pylog = logging.getLogger(__name__)


T = TypeVar("T")
POP_STRATEGIES = ("max", "min")
PopStrategy = Union[Literal["max", "min"], int]
VocabScores = TypedDict("VocabScores", {"vocab.cands": Tensor})
VocabOuts = tuple[VocabScores, VocabScores]


def vocab(
    candidates: list[str],
    mult_references: Union[list[list[str]], None],
    return_all_scores: bool = True,
    *,
    seed: Union[None, int, torch.Generator] = 1234,
    tokenizer: Callable[[str], list[str]] = str.split,
    dtype: torch.dtype = torch.float64,
    pop_strategy: PopStrategy = "max",
    verbose: int = 0,
) -> Union[VocabOuts, Tensor]:
    """Compute vocabulary statistics.

    Returns the candidate corpus vocabulary length, the references vocabulary length, the average vocabulary length for single references, and the vocabulary ratios between candidates and references.

    :param candidates: The list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target. Can also be None.
    :param return_all_scores: If True, returns a tuple containing the globals and locals scores.
        Otherwise returns a scalar tensor containing the main global score.
        defaults to True.
    :param seed: Random seed used to compute average vocabulary length for multiple references. defaults to 1234.
    :param tokenizer: The function used to split a sentence into tokens. defaults to str.split.
    :param dtype: Torch floating point dtype for numerical precision. defaults to torch.float64.
    :param pop_strategy: Strategy to compute average reference vocab. defaults to "max".
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple of globals and locals scores or a scalar tensor with the main global score.
    """
    if mult_references is not None:
        check_metric_inputs(candidates, mult_references)
    elif not is_mono_sents(candidates):
        error_msg = f"Invalid candidates type. (expected list[str], found {candidates.__class__.__name__})"
        raise ValueError(error_msg)

    tok_cands = list(map(tokenizer, candidates))
    del candidates

    vocab_cands_len = _corpus_vocab_size(tok_cands, dtype)
    _, vocab_per_cand = _sent_vocab_sizes(tok_cands, dtype)
    if not return_all_scores:
        return vocab_cands_len

    sents_scores = {
        "vocab.cands": vocab_per_cand,
    }
    corpus_scores = {
        "vocab.cands": vocab_cands_len,
    }

    if mult_references is None:
        vocab_outs = corpus_scores, sents_scores
        return vocab_outs  # type: ignore

    if len(mult_references) <= 0:
        msg = f"Invalid number of references. (found {len(mult_references)} references)"
        raise ValueError(msg)
    tok_mrefs = [list(map(tokenizer, refs)) for refs in mult_references]
    del mult_references

    corpus_vocab_cands = set(token for cand in tok_cands for token in cand)
    corpus_vocab_mrefs = set(
        token for refs in tok_mrefs for ref in refs for token in ref
    )

    inter = corpus_vocab_cands.intersection(corpus_vocab_mrefs)  # True positives
    diff = corpus_vocab_mrefs.difference(corpus_vocab_cands)  # False negatives
    union = corpus_vocab_cands.union(corpus_vocab_mrefs)

    vocab_precision = len(inter) / len(corpus_vocab_cands)
    vocab_recall = len(inter) / (len(inter) + len(diff))
    vocab_f1 = 2 * vocab_precision * vocab_recall / (vocab_precision + vocab_recall)
    vocab_jaccard = len(inter) / len(union)

    vocab_mrefs_len_full = _corpus_vocab_size(
        [ref for refs in tok_mrefs for ref in refs], dtype
    )
    vocab_ratio_len_full = vocab_cands_len / vocab_mrefs_len_full

    if isinstance(seed, int):
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = seed

    if pop_strategy == "max":
        num_try = max(len(refs) for refs in tok_mrefs)
    elif pop_strategy == "min":
        num_try = min(len(refs) for refs in tok_mrefs)
    elif isinstance(pop_strategy, int):
        num_try = pop_strategy
    else:
        msg = f"Invalid argument {pop_strategy=}. (expected one of {POP_STRATEGIES} or an integer value)"
        raise ValueError(msg)

    if verbose >= 2:
        pylog.debug(f"Found {num_try=} with {pop_strategy=}.")

    vocab_mrefs_lens = torch.empty((num_try,), dtype=dtype)

    for i in range(num_try):
        popped_refs, _ = _sample_sentences_split(tok_mrefs, generator=generator)
        vocab_mrefs_len_i = _corpus_vocab_size(popped_refs, dtype)
        vocab_mrefs_lens[i] = vocab_mrefs_len_i

    vocab_mrefs_avg = vocab_mrefs_lens.mean()
    vocab_len_ratio_avg = vocab_cands_len / vocab_mrefs_avg

    corpus_scores |= {
        "vocab.mrefs_full": vocab_mrefs_len_full,
        "vocab.ratio_full": vocab_ratio_len_full,
        "vocab.mrefs_avg": vocab_mrefs_avg,
        "vocab.ratio_avg": vocab_len_ratio_avg,
        "vocab.precision": vocab_precision,
        "vocab.recall": vocab_recall,
        "vocab.f1": vocab_f1,
        "vocab.jaccard": vocab_jaccard,
    }
    vocab_outs = corpus_scores, sents_scores
    return vocab_outs  # type: ignore


def _corpus_vocab_size(tok_sents: list[list[str]], dtype: torch.dtype) -> Tensor:
    corpus_vocab = set(token for sent in tok_sents for token in sent)
    vocab_len = torch.as_tensor(len(corpus_vocab), dtype=dtype)
    return vocab_len


def _sent_vocab_sizes(
    tok_sents: list[list[str]],
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    sents_cands_vocabs = [set(sent) for sent in tok_sents]
    sent_cands_vocabs_lens = torch.as_tensor(
        list(map(len, sents_cands_vocabs)), dtype=dtype
    )
    sent_cands_vocab_len = sent_cands_vocabs_lens.mean()
    return sent_cands_vocab_len, sent_cands_vocabs_lens


def _sample_sentences_split(
    mult_sentences: list[list[T]],
    generator: Union[Generator, None] = None,
) -> tuple[list[T], list[list[T]]]:
    candidates: list[T] = []
    mult_references: list[list[T]] = []

    for sents in mult_sentences:
        idx = int(torch.randint(0, len(sents), (), generator=generator).item())
        cand = sents[idx]
        refs = [sent for i, sent in enumerate(sents) if i != idx]
        candidates.append(cand)
        mult_references.append(refs)

    return candidates, mult_references
