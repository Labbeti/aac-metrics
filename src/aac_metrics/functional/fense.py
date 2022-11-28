#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BASED ON https://github.com/blmoistawinde/fense/
"""

import logging

from typing import Optional, Union

import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from torch import Tensor
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from aac_metrics.functional._fense_utils import (
    BERTFlatClassifier,
    infer_preprocess,
    load_pretrain_echecker,
)


pylog = logging.getLogger(__name__)


def fense(
    candidates: list[str],
    mult_references: list[list[str]],
    return_all_scores: bool = True,
    sbert_model: Union[str, SentenceTransformer] = "paraphrase-TinyBERT-L6-v2",
    echecker: Union[None, str, BERTFlatClassifier] = "echecker_clotho_audiocaps_base",
    echecker_tokenizer: Optional[AutoTokenizer] = None,
    error_threshold: float = 0.9,
    penalty: float = 0.9,
    agg_score: str = "mean",
    device: Union[str, torch.device, None] = "cpu",
    batch_size: int = 32,
    verbose: int = 0,
) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Tensor]]]:
    """Fluency ENhanced Sentence-bert Evaluation (FENSE)

    Paper: https://arxiv.org/abs/2110.04684
    Original implementation: https://github.com/blmoistawinde/fense

    :param candidates: The list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target.
    :param return_all_scores: If True, returns a tuple containing the globals and locals scores.
        Otherwise returns a scalar tensor containing the main global score.
        defaults to True.
    :param sbert_model: The sentence BERT model used to extract sentence embeddings for cosine-similarity. defaults to "paraphrase-TinyBERT-L6-v2".
    :param echecker: The echecker model used to detect fluency errors.
        Can be "echecker_clotho_audiocaps_base", "echecker_clotho_audiocaps_tiny", "none" or None.
        defaults to "echecker_clotho_audiocaps_base".
    :param echecker_tokenizer: The tokenizer of the echecker model.
        If None and echecker is not None, this value will be inferred with `echecker.model_type`.
        defaults to None.
    :param error_threshold: The threshold used to detect fluency errors for echecker model. defaults to 0.9.
    :param penalty: The penalty coefficient applied. Higher value means to lower the cos-sim scores when an error is detected. defaults to 0.9.
    :param agg_score: The aggregate function applied. Can be "mean", "max" or "sum". defaults to "mean".
    :param device: The pytorch device used to run the sBERT and echecker models. defaults to "cpu".
    :param batch_size: The batch size of the sBERT and echecker models. defaults to 32.
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple of globals and locals scores or a scalar tensor with the main global score.
    """

    # Init models
    sbert_model, echecker, echecker_tokenizer = _load_models_and_tokenizer(
        sbert_model, echecker, echecker_tokenizer
    )

    # Encode sents
    rng_ids = [0]
    for refs in mult_references:
        rng_ids.append(rng_ids[-1] + len(refs))
    flat_references = [ref for refs in mult_references for ref in refs]

    cands_embs = _encode_sents_sbert(sbert_model, candidates, batch_size, verbose)
    mrefs_embs = _encode_sents_sbert(sbert_model, flat_references, batch_size, verbose)

    # Compute sBERT similarities
    sbert_sim_scores = [
        (cands_embs[i] @ mrefs_embs[rng_ids[i] : rng_ids[i + 1]].T).mean().item()
        for i in range(len(cands_embs))
    ]
    sbert_sim_scores = np.array(sbert_sim_scores)

    # Compute fluency error detection penalty
    if echecker is not None and echecker_tokenizer is not None:
        has_error, _probs = _detect_error_sents(
            echecker,
            echecker_tokenizer,  # type: ignore
            candidates,
            error_threshold,
            batch_size,
            device,
        )
        fense_scores = sbert_sim_scores * (1.0 - penalty * has_error)
    else:
        has_error = None
        fense_scores = sbert_sim_scores

    # Aggregate and return
    if agg_score == "mean":
        reduction = np.mean
    elif agg_score == "max":
        reduction = np.max
    elif agg_score == "sum":
        reduction = np.sum
    else:
        AGG_SCORES = ("mean", "max", "sum")
        raise ValueError(
            f"Invalid argument {agg_score=}. (expected one of {AGG_SCORES})"
        )

    sbert_sim_score = reduction(sbert_sim_scores)
    fense_score = reduction(fense_scores)

    sbert_sim_score = torch.as_tensor(sbert_sim_score)
    fense_score = torch.as_tensor(fense_score)
    sbert_sim_scores = torch.from_numpy(sbert_sim_scores)
    fense_scores = torch.from_numpy(fense_scores)

    if return_all_scores:
        corpus_scores = {
            "fense": fense_score,
            "sbert_sim": sbert_sim_score,
        }
        sents_scores = {
            "fense": fense_scores,
            "sbert_sim": sbert_sim_scores,
        }

        if has_error is not None:
            error_rates = torch.from_numpy(has_error)
            error_rate = error_rates.mean()
            corpus_scores["fluency_error"] = error_rate
            sents_scores["fluency_error"] = error_rates

        return corpus_scores, sents_scores
    else:
        return fense_score


def _load_models_and_tokenizer(
    sbert_model: Union[str, SentenceTransformer] = "paraphrase-TinyBERT-L6-v2",
    echecker: Union[None, str, BERTFlatClassifier] = "echecker_clotho_audiocaps_base",
    echecker_tokenizer: Optional[AutoTokenizer] = None,
    device: Union[str, torch.device, None] = "cpu",
) -> tuple[SentenceTransformer, Optional[BERTFlatClassifier], Optional[AutoTokenizer]]:
    if isinstance(sbert_model, str):
        sbert_model = SentenceTransformer(sbert_model, device=device)  # type: ignore
    sbert_model.to(device)

    if isinstance(echecker, str):
        if echecker == "none":
            echecker = None
        else:
            echecker = load_pretrain_echecker(echecker, device)

    if echecker_tokenizer is None and echecker is not None:
        echecker_tokenizer = AutoTokenizer.from_pretrained(echecker.model_type)

    for model in (sbert_model, echecker):
        if model is None:
            continue
        for p in model.parameters():
            p.detach_()
        model.eval()

    return sbert_model, echecker, echecker_tokenizer


def _encode_sents_sbert(
    sbert_model: SentenceTransformer,
    sents: list[str],
    batch_size: int = 32,
    verbose: int = 0,
) -> Tensor:
    return sbert_model.encode(
        sents,
        convert_to_tensor=True,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=verbose >= 2,
    )  # type: ignore


def _detect_error_sents(
    echecker: BERTFlatClassifier,
    echecker_tokenizer: PreTrainedTokenizerFast,
    sents: list[str],
    error_threshold: float,
    batch_size: int,
    device: Union[str, torch.device, None],
    max_len: int = 64,
) -> tuple[np.ndarray, np.ndarray]:

    if len(sents) <= batch_size:
        batch = infer_preprocess(
            echecker_tokenizer,
            sents,
            max_len=max_len,
            device=device,
            dtype=torch.long,
        )
        logits = echecker(**batch)
        assert not logits.requires_grad
        # note: fix error in the original fense code: https://github.com/blmoistawinde/fense/blob/main/fense/evaluator.py#L69
        probs = torch.sigmoid(logits)[:, -1].cpu().numpy()

    else:
        probs_lst = []

        for i in range(0, len(sents), batch_size):
            batch = infer_preprocess(
                echecker_tokenizer,
                sents[i : i + batch_size],
                max_len=max_len,
                device=device,
                dtype=torch.long,
            )

            batch_logits = echecker(**batch)
            assert not batch_logits.requires_grad
            # batch_logits: (bsize, num_classes=6)
            batch_probs = torch.sigmoid(batch_logits)[:, -1].cpu().numpy()
            probs_lst.append(batch_probs)

        probs = np.concatenate(probs_lst)

    return (probs > error_threshold).astype(float), probs
