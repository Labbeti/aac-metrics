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
from transformers import logging as tfmers_logging
from transformers.models.auto.tokenization_auto import AutoTokenizer

from aac_metrics.functional._fense_utils import (
    BERTFlatClassifier,
    RemoteFileMetadata,
    _check_download_resource,
    _infer_preprocess,
)


PRETRAIN_ECHECKERS_DICT = {
    "echecker_clotho_audiocaps_base": (
        "https://github.com/blmoistawinde/fense/releases/download/V0.1/echecker_clotho_audiocaps_base.ckpt",
        "1a719f090af70614bbdb9f9437530b7e133c48cfa4a58d964de0d47fc974a2fa",
    ),
    "echecker_clotho_audiocaps_tiny": (
        "https://github.com/blmoistawinde/fense/releases/download/V0.1/echecker_clotho_audiocaps_tiny.ckpt",
        "90ed0ac5033ec497ec66d4f68588053813e085671136dae312097c96c504f673",
    ),
    "none": (None, None),
}
PRETRAIN_ECHECKERS = tuple(PRETRAIN_ECHECKERS_DICT.keys())


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
    device: Union[torch.device, str, None] = "cuda",
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
    :param device: The pytorch device used to run the sBERT and echecker models. defaults to "cuda".
    :param batch_size: The batch size of the sBERT and echecker models. defaults to 32.
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple of globals and locals scores or a scalar tensor with the main global score.
    """

    # Init models
    if isinstance(sbert_model, str):
        sbert_model = SentenceTransformer(sbert_model, device=device)  # type: ignore

    if isinstance(echecker, str):
        if echecker == "none":
            echecker = None
        else:
            echecker = _load_pretrain_echecker(echecker, device)

    if echecker_tokenizer is None and echecker is not None:
        echecker_tokenizer = AutoTokenizer.from_pretrained(echecker.model_type)

    for model in (sbert_model, echecker):
        if model is None:
            continue
        for p in model.parameters():
            p.detach_()
        model.eval()

    # Encode sents
    rng_ids = [0]
    flat_references = []
    for refs in mult_references:
        rng_ids.append(rng_ids[-1] + len(refs))
        flat_references.extend(refs)

    emb_cands = _encode_sents_sbert(sbert_model, candidates, batch_size, verbose)
    emb_refs = _encode_sents_sbert(sbert_model, flat_references, batch_size, verbose)

    # Compute sBERT similarities
    sim_scores = [
        (emb_cands[i] @ emb_refs[rng_ids[i] : rng_ids[i + 1]].T).mean().item()
        for i in range(len(candidates))
    ]
    sim_scores = np.array(sim_scores)

    # Compute fluency error detection penalty
    if echecker is not None and echecker_tokenizer is not None:
        has_error, probs = _detect_error_sents(
            echecker,
            echecker_tokenizer,
            candidates,
            error_threshold,
            batch_size,
            device,
        )
        fense_scores = sim_scores * (1.0 - penalty * has_error)
    else:
        has_error = None
        fense_scores = sim_scores

    # Aggregate and return
    if agg_score == "mean":
        fense_score = fense_scores.mean()
    elif agg_score == "max":
        fense_score = fense_scores.max()
    elif agg_score == "sum":
        fense_score = fense_scores.sum()
    else:
        raise ValueError(f"Invalid argument {agg_score=}.")

    fense_score = torch.as_tensor(fense_score)
    fense_scores = torch.from_numpy(fense_scores)

    if return_all_scores:
        global_scores = {
            "fense_score": fense_score,
        }
        local_scores = {
            "fense_score": fense_scores,
        }

        if has_error is not None:
            error_rates = torch.from_numpy(has_error)
            error_rate = error_rates.mean()
            global_scores["fense_error_rate"] = error_rate
            global_scores["fense_error_rate"] = error_rates

        return global_scores, local_scores
    else:
        return fense_score


def _load_pretrain_echecker(
    echecker_model: str,
    device: Union[torch.device, str, None] = "cuda",
    use_proxy: bool = False,
    proxies: Optional[dict[str, str]] = None,
) -> BERTFlatClassifier:
    tfmers_logging.set_verbosity_error()  # suppress loading warnings
    url, checksum = PRETRAIN_ECHECKERS_DICT[echecker_model]
    remote = RemoteFileMetadata(
        filename=f"{echecker_model}.ckpt", url=url, checksum=checksum
    )
    file_path = _check_download_resource(remote, use_proxy, proxies)
    model_states = torch.load(file_path)
    echecker = BERTFlatClassifier(
        model_type=model_states["model_type"],
        num_classes=model_states["num_classes"],
    )
    echecker.load_state_dict(model_states["state_dict"])
    echecker.eval()
    echecker.to(device)
    return echecker


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
    echecker_tokenizer: AutoTokenizer,
    sents: list[str],
    error_threshold: float,
    batch_size: int,
    device: Union[torch.device, str, None],
) -> tuple[np.ndarray, np.ndarray]:

    if len(sents) <= batch_size:
        batch = _infer_preprocess(echecker_tokenizer, sents, max_len=64)
        for k, v in batch.items():
            batch[k] = v.to(device)

        logits = echecker(**batch)
        assert not logits.requires_grad
        probs = torch.sigmoid(logits).cpu().numpy()
    else:
        probs = []
        for i in range(0, len(sents), batch_size):
            batch = _infer_preprocess(
                echecker_tokenizer, sents[i : i + batch_size], max_len=64
            )
            for k, v in batch.items():
                batch[k] = v.to(device)

            batch_logits = echecker(**batch)
            assert not batch_logits.requires_grad
            batch_probs = torch.sigmoid(batch_logits)[:, -1].cpu().numpy()
            probs.append(batch_probs)

        probs = np.concatenate(probs)
    return (probs > error_threshold).astype(float), probs
