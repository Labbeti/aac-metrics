#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BASED ON https://github.com/blmoistawinde/fense/
"""

import logging

from typing import Union

import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from torch import Tensor


pylog = logging.getLogger(__name__)


def sbert(
    candidates: list[str],
    mult_references: list[list[str]],
    return_all_scores: bool = True,
    sbert_model: Union[str, SentenceTransformer] = "paraphrase-TinyBERT-L6-v2",
    device: Union[str, torch.device] = "auto",
    batch_size: int = 32,
    verbose: int = 0,
) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Tensor]]]:
    # Init models
    sbert_model = _load_model(sbert_model, device)

    # Encode sents
    rng_ids = [0]
    for refs in mult_references:
        rng_ids.append(rng_ids[-1] + len(refs))
    flat_references = [ref for refs in mult_references for ref in refs]

    cands_embs = _encode_sents_sbert(sbert_model, candidates, batch_size, verbose)
    mrefs_embs = _encode_sents_sbert(sbert_model, flat_references, batch_size, verbose)

    # Compute sBERT similarities
    sbert_cos_sims = [
        (cands_embs[i] @ mrefs_embs[rng_ids[i] : rng_ids[i + 1]].T).mean().item()
        for i in range(len(cands_embs))
    ]
    sbert_cos_sims = np.array(sbert_cos_sims)

    # Aggregate and return
    sbert_cos_sim = sbert_cos_sims.mean()

    sbert_cos_sim = torch.as_tensor(sbert_cos_sim)
    sbert_cos_sims = torch.from_numpy(sbert_cos_sims)

    if return_all_scores:
        sents_scores = {
            "sbert.sim": sbert_cos_sims,
        }
        corpus_scores = {
            "sbert.sim": sbert_cos_sim,
        }

        return corpus_scores, sents_scores
    else:
        return sbert_cos_sim


def _load_model(
    sbert_model: Union[str, SentenceTransformer] = "paraphrase-TinyBERT-L6-v2",
    device: Union[str, torch.device] = "auto",
) -> SentenceTransformer:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, str):
        device = torch.device(device)

    if isinstance(sbert_model, str):
        sbert_model = SentenceTransformer(sbert_model, device=device)  # type: ignore
    sbert_model.to(device)

    for p in sbert_model.parameters():
        p.detach_()
    sbert_model.eval()

    return sbert_model


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
