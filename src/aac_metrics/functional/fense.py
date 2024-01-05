#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Optional, Union

import torch

from sentence_transformers import SentenceTransformer
from torch import Tensor
from transformers.models.auto.tokenization_auto import AutoTokenizer

from aac_metrics.functional.fer import (
    fer,
    _load_echecker_and_tokenizer,
    BERTFlatClassifier,
    DEFAULT_FER_MODEL,
)
from aac_metrics.functional.sbert_sim import (
    sbert_sim,
    _load_sbert,
    DEFAULT_SBERT_SIM_MODEL,
)
from aac_metrics.utils.checks import check_metric_inputs


pylog = logging.getLogger(__name__)


def fense(
    candidates: list[str],
    mult_references: list[list[str]],
    return_all_scores: bool = True,
    # SBERT args
    sbert_model: Union[str, SentenceTransformer] = DEFAULT_SBERT_SIM_MODEL,
    # FluencyError args
    echecker: Union[str, BERTFlatClassifier] = DEFAULT_FER_MODEL,
    echecker_tokenizer: Optional[AutoTokenizer] = None,
    error_threshold: float = 0.9,
    device: Union[str, torch.device, None] = "cuda_if_available",
    batch_size: int = 32,
    reset_state: bool = True,
    return_probs: bool = False,
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
    :param device: The PyTorch device used to run FENSE models. If "cuda_if_available", it will use cuda if available. defaults to "cuda_if_available".
    :param batch_size: The batch size of the sBERT and echecker models. defaults to 32.
    :param reset_state: If True, reset the state of the PyTorch global generator after the initialization of the pre-trained models. defaults to True.
    :param return_probs: If True, return each individual error probability given by the fluency detector model. defaults to False.
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple of globals and locals scores or a scalar tensor with the main global score.
    """
    check_metric_inputs(candidates, mult_references)

    # Init models
    sbert_model, echecker, echecker_tokenizer = _load_models_and_tokenizer(
        sbert_model=sbert_model,
        echecker=echecker,
        echecker_tokenizer=echecker_tokenizer,
        device=device,
        reset_state=reset_state,
        verbose=verbose,
    )
    sbert_sim_outs: tuple[dict[str, Tensor], dict[str, Tensor]] = sbert_sim(  # type: ignore
        candidates=candidates,
        mult_references=mult_references,
        return_all_scores=True,
        sbert_model=sbert_model,
        device=device,
        batch_size=batch_size,
        reset_state=reset_state,
        verbose=verbose,
    )
    fer_outs: tuple[dict[str, Tensor], dict[str, Tensor]] = fer(  # type: ignore
        candidates=candidates,
        return_all_scores=True,
        echecker=echecker,
        echecker_tokenizer=echecker_tokenizer,
        error_threshold=error_threshold,
        device=device,
        batch_size=batch_size,
        reset_state=reset_state,
        return_probs=return_probs,
        verbose=verbose,
    )
    fense_outs = _fense_from_outputs(sbert_sim_outs, fer_outs, penalty)

    if return_all_scores:
        return fense_outs
    else:
        return fense_outs[0]["fense"]


def _fense_from_outputs(
    sbert_sim_outs: tuple[dict[str, Tensor], dict[str, Tensor]],
    fer_outs: tuple[dict[str, Tensor], dict[str, Tensor]],
    penalty: float = 0.9,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Combines SBERT and FER outputs.

    Based on https://github.com/blmoistawinde/fense/blob/main/fense/evaluator.py#L121
    """
    sbert_sim_outs_corpus, sbert_sim_outs_sents = sbert_sim_outs
    fer_outs_corpus, fer_outs_sents = fer_outs

    sbert_sims_scores = sbert_sim_outs_sents["sbert_sim"]
    fer_scores = fer_outs_sents["fer"]
    fense_scores = sbert_sims_scores * (1.0 - penalty * fer_scores)
    fense_score = torch.as_tensor(
        fense_scores.cpu()
        .numpy()
        .mean(),  # note: use numpy mean to keep the same values than the original fense
        device=fense_scores.device,
    )

    fense_outs_corpus = sbert_sim_outs_corpus | fer_outs_corpus | {"fense": fense_score}
    fense_outs_sents = sbert_sim_outs_sents | fer_outs_sents | {"fense": fense_scores}
    fense_outs = fense_outs_corpus, fense_outs_sents

    return fense_outs


def _load_models_and_tokenizer(
    sbert_model: Union[str, SentenceTransformer] = DEFAULT_SBERT_SIM_MODEL,
    echecker: Union[str, BERTFlatClassifier] = DEFAULT_FER_MODEL,
    echecker_tokenizer: Optional[AutoTokenizer] = None,
    device: Union[str, torch.device, None] = "cuda_if_available",
    reset_state: bool = True,
    verbose: int = 0,
) -> tuple[SentenceTransformer, BERTFlatClassifier, AutoTokenizer]:
    sbert_model = _load_sbert(
        sbert_model=sbert_model,
        device=device,
        reset_state=reset_state,
    )
    echecker, echecker_tokenizer = _load_echecker_and_tokenizer(
        echecker=echecker,
        echecker_tokenizer=echecker_tokenizer,
        device=device,
        reset_state=reset_state,
        verbose=verbose,
    )
    return sbert_model, echecker, echecker_tokenizer
