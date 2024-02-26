#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Union

import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from torch import Tensor

from aac_metrics.utils.checks import check_metric_inputs
from aac_metrics.utils.globals import _get_device


DEFAULT_SBERT_SIM_MODEL = "paraphrase-TinyBERT-L6-v2"

pylog = logging.getLogger(__name__)


def sbert_sim(
    candidates: list[str],
    mult_references: list[list[str]],
    return_all_scores: bool = True,
    sbert_model: Union[str, SentenceTransformer] = DEFAULT_SBERT_SIM_MODEL,
    device: Union[str, torch.device, None] = "cuda_if_available",
    batch_size: int = 32,
    reset_state: bool = True,
    verbose: int = 0,
) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Tensor]]]:
    """Cosine-similarity of the Sentence-BERT embeddings.

    - Paper: https://arxiv.org/abs/1908.10084
    - Original implementation: https://github.com/blmoistawinde/fense

    :param candidates: The list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target.
    :param return_all_scores: If True, returns a tuple containing the globals and locals scores.
        Otherwise returns a scalar tensor containing the main global score.
        defaults to True.
    :param sbert_model: The sentence BERT model used to extract sentence embeddings for cosine-similarity. defaults to "paraphrase-TinyBERT-L6-v2".
    :param device: The PyTorch device used to run FENSE models. If "cuda_if_available", it will use cuda if available. defaults to "cuda_if_available".
    :param batch_size: The batch size of the sBERT models. defaults to 32.
    :param reset_state: If True, reset the state of the PyTorch global generator after the initialization of the pre-trained models. defaults to True.
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple of globals and locals scores or a scalar tensor with the main global score.
    """
    check_metric_inputs(candidates, mult_references)

    # Init models
    sbert_model = _load_sbert(sbert_model, device, reset_state)

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

    # Aggregate and return
    sbert_sim_score = sbert_sim_scores.mean()

    sbert_sim_score = torch.as_tensor(sbert_sim_score)
    sbert_sim_scores = torch.from_numpy(sbert_sim_scores)

    if return_all_scores:
        sbert_sim_outs_corpus = {
            "sbert_sim": sbert_sim_score,
        }
        sbert_sim_outs_sents = {
            "sbert_sim": sbert_sim_scores,
        }

        return sbert_sim_outs_corpus, sbert_sim_outs_sents
    else:
        return sbert_sim_score


def _load_sbert(
    sbert_model: Union[str, SentenceTransformer] = DEFAULT_SBERT_SIM_MODEL,
    device: Union[str, torch.device, None] = "cuda_if_available",
    reset_state: bool = True,
) -> SentenceTransformer:
    state = torch.random.get_rng_state()

    device = _get_device(device)
    if isinstance(sbert_model, str):
        sbert_model = SentenceTransformer(sbert_model, device=device)  # type: ignore
    sbert_model.to(device=device)

    sbert_model = sbert_model.eval()
    for p in sbert_model.parameters():
        p.requires_grad_(False)

    if reset_state:
        torch.random.set_rng_state(state)
    return sbert_model


@torch.no_grad()
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
