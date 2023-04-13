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

from aac_metrics.functional.fluerr import (
    fluerr,
    _load_echecker_and_tokenizer,
    BERTFlatClassifier,
)
from aac_metrics.functional.sbert_sim import sbert_sim, _load_sbert


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
    return_probs: bool = True,
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
    :param device: The PyTorch device used to run FENSE models. If "auto", it will use cuda if available. defaults to "auto".
    :param batch_size: The batch size of the sBERT and echecker models. defaults to 32.
    :param reset_state: If True, reset the state of the PyTorch global generator after the pre-trained model are built. defaults to True.
    :param return_probs: If True, return each individual error probability given by the fluency detector model. defaults to True.
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple of globals and locals scores or a scalar tensor with the main global score.
    """

    # Init models
    sbert_model, echecker, echecker_tokenizer = _load_models_and_tokenizer(
        sbert_model, echecker, echecker_tokenizer, device, reset_state, verbose
    )

    sbert_sim_outs: tuple = sbert_sim(candidates, mult_references, True, sbert_model, device, batch_size, reset_state, verbose)  # type: ignore
    fluerr_outs: tuple = fluerr(candidates, True, echecker, echecker_tokenizer, error_threshold, device, batch_size, reset_state, return_probs, verbose)  # type: ignore
    fense_outs = _fense_from_outputs(sbert_sim_outs, fluerr_outs, penalty)

    if return_all_scores:
        return fense_outs
    else:
        return fense_outs[0]["fense"]


def _fense_from_outputs(
    sbert_sim_outs: tuple[dict[str, Tensor], dict[str, Tensor]],
    fluerr_outs: tuple[dict[str, Tensor], dict[str, Tensor]],
    penalty: float,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Combines SBERT and FluErr outputs.

    Based on https://github.com/blmoistawinde/fense/blob/main/fense/evaluator.py#L121
    """
    sbert_sim_outs_corpus, sbert_sim_outs_sents = sbert_sim_outs
    fluerr_outs_corpus, fluerr_outs_sents = fluerr_outs

    sbert_sims_scores = sbert_sim_outs_sents["sbert_sim"]
    fluerr_scores = fluerr_outs_sents["fluerr"]
    fense_scores = sbert_sims_scores * (1.0 - penalty * fluerr_scores)
    fense_score = torch.as_tensor(
        fense_scores.cpu().numpy().mean(),
        device=fense_scores.device,
    )

    fense_outs_corpus = (
        sbert_sim_outs_corpus | fluerr_outs_corpus | {"fense": fense_score}
    )
    fense_outs_sents = (
        sbert_sim_outs_sents | fluerr_outs_sents | {"fense": fense_scores}
    )
    fense_outs = fense_outs_corpus, fense_outs_sents

    return fense_outs


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
