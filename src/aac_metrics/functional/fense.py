#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FENSE metric functional API.

Based on original implementation in https://github.com/blmoistawinde/fense/
"""

import logging

from typing import Optional, Union

import torch

from sentence_transformers import SentenceTransformer
from torch import Tensor
from transformers.models.auto.tokenization_auto import AutoTokenizer

from aac_metrics.functional.fluency_error import (
    fluency_error,
    _load_echecker_and_tokenizer,
    BERTFlatClassifier,
)
from aac_metrics.functional.sbert import sbert, _load_sbert


pylog = logging.getLogger(__name__)


def fense(
    candidates: list[str],
    mult_references: list[list[str]],
    return_all_scores: bool = True,
    # SBERT args
    sbert_model: Union[str, SentenceTransformer] = "paraphrase-TinyBERT-L6-v2",
    # FluencyError args
    echecker: Union[str, BERTFlatClassifier] = "echecker_clotho_audiocaps_base",
    echecker_tokenizer: Optional[AutoTokenizer] = None,
    error_threshold: float = 0.9,
    device: Union[str, torch.device, None] = "auto",
    batch_size: int = 32,
    reset_state: bool = True,
    # Other args
    penalty: float = 0.9,
    verbose: int = 0,
) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Tensor]]]:
    """Fluency ENhanced Sentence-bert Evaluation (FENSE)

    - Paper: https://arxiv.org/abs/2110.04684
    - Original implementation: https://github.com/blmoistawinde/fense

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
    :param device: The PyTorch device used to run FENSE models. If "auto", it will use cuda if available. defaults to "cpu".
    :param batch_size: The batch size of the sBERT and echecker models. defaults to 32.
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple of globals and locals scores or a scalar tensor with the main global score.
    """

    # Init models
    sbert_model, echecker, echecker_tokenizer = _load_models_and_tokenizer(
        sbert_model, echecker, echecker_tokenizer, device, reset_state, verbose
    )

    sbert_corpus_scores, sbert_sents_scores = sbert(candidates, mult_references, True, sbert_model, device, batch_size, verbose)  # type: ignore
    sbert_corpus_scores: dict[str, Tensor]
    sbert_sents_scores: dict[str, Tensor]

    fluerr_corpus_scores, fluerr_sents_scores = fluency_error(candidates, True, echecker, echecker_tokenizer, error_threshold, device, batch_size, verbose)  # type: ignore
    fluerr_corpus_scores: dict[str, Tensor]
    fluerr_sents_scores: dict[str, Tensor]

    sbert_cos_sims = sbert_sents_scores["sbert.sim"]
    fluency_errors = fluerr_sents_scores["fluency_error"]
    fense_scores = sbert_cos_sims * (1.0 - penalty * fluency_errors)
    fense_score = fense_scores.mean()

    if return_all_scores:
        corpus_scores = (
            sbert_corpus_scores | fluerr_corpus_scores | {"fense": fense_score}
        )
        sents_scores = (
            sbert_sents_scores | fluerr_sents_scores | {"fense": fense_scores}
        )
        return corpus_scores, sents_scores
    else:
        return fense_score


def _load_models_and_tokenizer(
    sbert_model: Union[str, SentenceTransformer] = "paraphrase-TinyBERT-L6-v2",
    echecker: Union[str, BERTFlatClassifier] = "echecker_clotho_audiocaps_base",
    echecker_tokenizer: Optional[AutoTokenizer] = None,
    device: Union[str, torch.device, None] = "auto",
    reset_state: bool = True,
    verbose: int = 0,
) -> tuple[SentenceTransformer, BERTFlatClassifier, AutoTokenizer]:
    sbert_model = _load_sbert(sbert_model, device, reset_state)
    echecker, echecker_tokenizer = _load_echecker_and_tokenizer(
        echecker, echecker_tokenizer, device, reset_state, verbose
    )
    return sbert_model, echecker, echecker_tokenizer
