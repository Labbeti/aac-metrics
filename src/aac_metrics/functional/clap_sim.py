#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import random
from typing import Literal, Optional, TypedDict, Union

import numpy as np
import torch
from msclap import CLAP
from torch import Tensor
from torch.nn import CosineSimilarity

from aac_metrics.utils.globals import _get_device

pylog = logging.getLogger(__name__)

DEFAULT_CLAP_SIM_MODEL = "MS-CLAP-2023"

CLAPMethod = Literal["audio", "text"]
CLAP_METHODS = ("audio", "text")

CLAPScores = TypedDict("CLAPScores", {"clap_sim": Tensor})
CLAPOuts = tuple[CLAPScores, CLAPScores]


def clap_sim(
    candidates: list[str],
    mult_references: Optional[list[list[str]]] = None,
    audio_paths: Optional[list[str]] = None,
    return_all_scores: bool = True,
    *,
    clap_method: CLAPMethod = "text",
    clap_model: Union[str, CLAP] = DEFAULT_CLAP_SIM_MODEL,
    device: Union[str, torch.device, None] = "cuda_if_available",
    batch_size: Optional[int] = 32,
    reset_state: bool = True,
    seed: Optional[int] = 42,
    verbose: int = 0,
) -> Union[Tensor, CLAPOuts]:
    """Cosine-similarity of the Contrastive Language-Audio Pretraining (CLAP) embeddings.

    The implementation is based on the msclap pypi package.

    - Paper: https://arxiv.org/pdf/2411.00321
    - msclap package: https://pypi.org/project/msclap/

    :param candidates: The list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target when method is "text". defaults to None.
    :param audio_paths: Audio filepaths required when method is "audio". defaults to None.
    :param return_all_scores: If True, returns a tuple containing the globals and locals scores.
        Otherwise returns a scalar tensor containing the main global score.
        defaults to True.
    :param clap_method: The method used to encode the sentences. Can be "text" or "audio". defaults to "text".
    :param clap_model: The CLAP model used to extract sentence embeddings for cosine-similarity. defaults to "2023".
    :param device: The PyTorch device used to run MACE models. If "cuda_if_available", it will use cuda if available. defaults to "cuda_if_available".
    :param batch_size: The batch size of the CLAP models. defaults to 32.
    :param reset_state: If True, reset the state of the PyTorch global generator after the initialization of the pre-trained models. defaults to True.
    :param seed: Optional seed to make CLAP-sim scores deterministic when using clap_method="audio" on large audio files. defaults to 42.
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple of globals and locals scores or a scalar tensor with the main global score.
    """
    # Init models
    clap_model = _load_clap(clap_model, device, reset_state)
    cands_embs = _encode_sents_clap(clap_model, candidates, batch_size)

    if clap_method == "text":
        if mult_references is None:
            msg = f"Invalid arguments {clap_method=} with {mult_references=}."
            raise ValueError(msg)

        rng_ids = [0]
        for refs in mult_references:
            rng_ids.append(rng_ids[-1] + len(refs))
        flat_references = [ref for refs in mult_references for ref in refs]
        mrefs_embs = _encode_sents_clap(clap_model, flat_references, batch_size)

    elif clap_method == "audio":
        if audio_paths is None:
            msg = f"Invalid arguments {clap_method=} with {audio_paths=}."
            raise ValueError(msg)

        rng_ids = [i for i in range(len(audio_paths) + 1)]
        if seed is not None:
            random.seed(seed)
        mrefs_embs = _encode_audios_clap(clap_model, audio_paths, batch_size)

    else:
        msg = f"Invalid argument {clap_method=}. (expected one of {CLAP_METHODS})"
        raise ValueError(msg)

    clap_sim_scores = [
        _cosine_similarity(cands_embs[i], mrefs_embs[rng_ids[i] : rng_ids[i + 1]])
        .mean()
        .item()
        for i in range(len(cands_embs))
    ]
    clap_sim_scores = np.array(clap_sim_scores)

    # Aggregate and return
    clap_sim_score = clap_sim_scores.mean()
    clap_sim_score = torch.as_tensor(clap_sim_score)
    clap_sim_scores = torch.as_tensor(clap_sim_scores)

    if return_all_scores:
        clap_sim_outs_corpus = {
            "clap_sim": clap_sim_score,
        }
        clap_sim_outs_sents = {
            "clap_sim": clap_sim_scores,
        }
        clap_outs = clap_sim_outs_corpus, clap_sim_outs_sents
        return clap_outs  # type: ignore
    else:
        return clap_sim_score


def _cosine_similarity(input: Tensor, target: Tensor) -> Tensor:
    cos = CosineSimilarity(dim=-1, eps=1e-6)
    return cos(input.unsqueeze(0), target)


def _load_clap(
    clap_model: Union[str, CLAP] = DEFAULT_CLAP_SIM_MODEL,
    device: Union[str, torch.device, None] = "cuda_if_available",
    reset_state: bool = True,
) -> CLAP:
    state = torch.random.get_rng_state()

    device = _get_device(device)
    if isinstance(clap_model, str):
        use_cuda = device is not None and device.type == "cuda"
        clap_model = CLAP(version="2023", use_cuda=use_cuda)

    if reset_state:
        torch.random.set_rng_state(state)
    return clap_model


@torch.no_grad()
def _encode_sents_clap(
    clap_model: CLAP,
    sents: list[str],
    batch_size: Optional[int] = 32,
) -> Tensor:
    if batch_size is None:
        batch_size = len(sents)

    clap_embeddings = []
    for i in range(0, len(sents), batch_size):
        sents_batch = sents[i : i + batch_size]
        clap_embeddings_batch = clap_model.get_text_embeddings(sents_batch)
        clap_embeddings.append(clap_embeddings_batch)
    clap_embeddings = torch.vstack(clap_embeddings)
    return clap_embeddings


@torch.no_grad()
def _encode_audios_clap(
    clap_model: CLAP,
    audio_paths: list[str],
    batch_size: Optional[int] = 32,
) -> Tensor:
    audio_paths = list(map(str, audio_paths))
    if batch_size is None:
        batch_size = len(audio_paths)

    clap_embeddings = []
    for i in range(0, len(audio_paths), batch_size):
        audio_paths_batch = audio_paths[i : i + batch_size]
        clap_embeddings_batch = clap_model.get_audio_embeddings(audio_paths_batch)
        clap_embeddings.append(clap_embeddings_batch)
    clap_embeddings = torch.vstack(clap_embeddings)
    return clap_embeddings
