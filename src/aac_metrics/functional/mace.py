#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Any, Literal, MutableMapping, Optional, TypedDict, Union

import torch
from msclap import CLAP
from torch import Tensor
from transformers.models.auto.tokenization_auto import AutoTokenizer

from aac_metrics.functional.clap_sim import (
    DEFAULT_CLAP_SIM_MODEL,
    CLAPOuts,
    _load_clap,
    clap_sim,
)
from aac_metrics.functional.fer import (
    DEFAULT_FER_MODEL,
    BERTFlatClassifier,
    FEROuts,
    _load_echecker_and_tokenizer,
    fer,
)

pylog = logging.getLogger(__name__)

MACEMethod = Literal["text", "audio", "combined"]
MACE_METHODS = ("text", "audio", "combined")

MACEScores = TypedDict(
    "MACEScores", {"mace": Tensor, "fer": Tensor, "clap_sim": Tensor}
)
MACEOuts = tuple[MACEScores, MACEScores]


def mace(
    candidates: list[str],
    mult_references: Optional[list[list[str]]] = None,
    audio_paths: Optional[list[str]] = None,
    return_all_scores: bool = True,
    *,
    mace_method: MACEMethod = "text",
    penalty: float = 0.3,
    # CLAP args
    clap_model: Union[str, CLAP] = DEFAULT_CLAP_SIM_MODEL,
    seed: Optional[int] = 42,
    # FER args
    echecker: Union[str, BERTFlatClassifier] = DEFAULT_FER_MODEL,
    echecker_tokenizer: Optional[AutoTokenizer] = None,
    error_threshold: float = 0.97,
    device: Union[str, torch.device, None] = "cuda_if_available",
    batch_size: Optional[int] = 32,
    reset_state: bool = True,
    return_probs: bool = False,
    # Other args
    verbose: int = 0,
) -> Union[Tensor, MACEOuts]:
    """Multimodal Audio-Caption Evaluation class (MACE).

    MACE is a metric designed for evaluating automated audio captioning (AAC) systems.
    Unlike metrics that compare machine-generated captions solely to human references, MACE uses both audio and text to improve evaluation.
    By integrating both audio and text, it produces assessments that align better with human judgments.

    The implementation is based on the mace original implementation (original author have accepted to include their code in aac-metrics under the MIT license).

    - Paper: https://arxiv.org/pdf/2411.00321
    - Original author: Satvik Dixit
    - Original implementation: https://github.com/satvik-dixit/mace/tree/main

    :param candidates: The list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target when method is "text" or "combined". defaults to None.
    :param audio_paths: Audio filepaths required when method is "audio" or "combined". defaults to None.
    :param return_all_scores: If True, returns a tuple containing the globals and locals scores.
        Otherwise returns a scalar tensor containing the main global score.
        defaults to True.
    :param mace_method: The method used to encode the sentences. Can be "text", "audio" or "combined". defaults to "text".
    :param penalty: The penalty coefficient applied. Higher value means to lower the cos-sim scores when an error is detected. defaults to 0.3.
    :param clap_model: The CLAP model used to extract CLAP embeddings for cosine-similarity. defaults to "MS-CLAP-2023".
    :param seed: Optional seed to make CLAP-sim scores deterministic when using mace_method="audio" or "combined" on large audio files. defaults to 42.
    :param echecker: The echecker model used to detect fluency errors.
        Can be "echecker_clotho_audiocaps_base", "echecker_clotho_audiocaps_tiny", "none" or None.
        defaults to "echecker_clotho_audiocaps_base".
    :param echecker_tokenizer: The tokenizer of the echecker model.
        If None and echecker is not None, this value will be inferred with `echecker.model_type`.
        defaults to None.
    :param error_threshold: The threshold used to detect fluency errors for echecker model. defaults to 0.97.
    :param device: The PyTorch device used to run pre-trained models. If "cuda_if_available", it will use cuda if available. defaults to "cuda_if_available".
    :param batch_size: The batch size of the CLAP and echecker models. defaults to 32.
    :param reset_state: If True, reset the state of the PyTorch global generator after the initialization of the pre-trained models. defaults to True.
    :param return_probs: If True, return each individual error probability given by the fluency detector model. defaults to False.
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple of globals and locals scores or a scalar tensor with the main global score.
    """
    # Init models
    clap_model, echecker, echecker_tokenizer = _load_models_and_tokenizer(
        clap_model=clap_model,
        echecker=echecker,
        echecker_tokenizer=echecker_tokenizer,
        device=device,
        reset_state=reset_state,
        verbose=verbose,
    )

    clap_kwds: dict[str, Any] = dict(
        candidates=candidates,
        mult_references=mult_references,
        audio_paths=audio_paths,
        return_all_scores=True,
        clap_model=clap_model,
        device=device,
        batch_size=batch_size,
        reset_state=reset_state,
        seed=seed,
        verbose=verbose,
    )
    if mace_method == "text":
        clap_sim_outs: CLAPOuts = clap_sim(clap_method=mace_method, **clap_kwds)  # type: ignore

    elif mace_method == "audio":
        clap_sim_outs: CLAPOuts = clap_sim(clap_method=mace_method, **clap_kwds)  # type: ignore

    elif mace_method == "combined":
        clap_sim_outs_text: CLAPOuts = clap_sim(  # type: ignore
            clap_method="text", **clap_kwds
        )
        clap_sim_outs_audio: CLAPOuts = clap_sim(  # type: ignore
            clap_method="audio", **clap_kwds
        )

        clap_sim_outs_corpus_text, clap_sim_outs_sents_text = clap_sim_outs_text
        clap_sim_outs_corpus_audio, clap_sim_outs_sents_audio = clap_sim_outs_audio
        clap_sim_outs_corpus = _average_dicts(
            clap_sim_outs_corpus_text, clap_sim_outs_corpus_audio  # type: ignore
        )
        clap_sim_outs_sents = _average_dicts(
            clap_sim_outs_sents_text, clap_sim_outs_sents_audio  # type: ignore
        )
        clap_sim_outs: CLAPOuts = clap_sim_outs_corpus, clap_sim_outs_sents  # type: ignore

    else:
        msg = f"Invalid argument {mace_method=}. (expected one of {MACE_METHODS})"
        raise ValueError(msg)

    fer_outs: FEROuts = fer(  # type: ignore
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
    mace_outs = _mace_from_outputs(clap_sim_outs, fer_outs, penalty)

    if return_all_scores:
        return mace_outs
    else:
        return mace_outs[0]["mace"]


def _average_dicts(
    dict1: MutableMapping[str, Tensor],
    dict2: MutableMapping[str, Tensor],
) -> dict[str, Tensor]:
    averaged_dict = {}
    for key in dict1:
        averaged_dict[key] = (dict1[key] + dict2[key]) / 2
    return averaged_dict  # type: ignore


def _mace_from_outputs(
    clap_sim_outs: CLAPOuts,
    fer_outs: FEROuts,
    penalty: float = 0.3,
) -> MACEOuts:
    """Combines CLAP and FER outputs.

    Based on https://github.com/blmoistawinde/fense/blob/main/fense/evaluator.py#L121
    """
    clap_sim_outs_corpus, clap_sim_outs_sents = clap_sim_outs
    fer_outs_corpus, fer_outs_sents = fer_outs

    clap_sims_scores = clap_sim_outs_sents["clap_sim"]
    fer_scores = fer_outs_sents["fer"]
    mace_scores = clap_sims_scores * (1.0 - penalty * fer_scores)
    # note: we use numpy mean to keep the same values than the original mace, this is only for backward compatibility
    mace_score = torch.as_tensor(
        mace_scores.cpu().numpy().mean(),
        device=mace_scores.device,
    )

    mace_outs_corpus = clap_sim_outs_corpus | fer_outs_corpus | {"mace": mace_score}  # type: ignore
    mace_outs_sents = clap_sim_outs_sents | fer_outs_sents | {"mace": mace_score}  # type: ignore
    mace_outs = mace_outs_corpus, mace_outs_sents

    return mace_outs


def _load_models_and_tokenizer(
    clap_model: Union[str, CLAP] = DEFAULT_CLAP_SIM_MODEL,
    echecker: Union[str, BERTFlatClassifier] = DEFAULT_FER_MODEL,
    echecker_tokenizer: Optional[AutoTokenizer] = None,
    device: Union[str, torch.device, None] = "cuda_if_available",
    reset_state: bool = True,
    verbose: int = 0,
) -> tuple[CLAP, BERTFlatClassifier, AutoTokenizer]:
    clap_model = _load_clap(
        clap_model=clap_model,
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
    return clap_model, echecker, echecker_tokenizer
