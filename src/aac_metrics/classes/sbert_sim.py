#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Union

import torch

from sentence_transformers import SentenceTransformer
from torch import Tensor

from aac_metrics.classes.base import AACMetric
from aac_metrics.functional.sbert_sim import (
    sbert_sim,
    _load_sbert,
    DEFAULT_SBERT_SIM_MODEL,
)
from aac_metrics.utils.globals import _get_device


pylog = logging.getLogger(__name__)


class SBERTSim(AACMetric[Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]]):
    """Cosine-similarity of the Sentence-BERT embeddings.

    - Paper: https://arxiv.org/abs/1908.10084
    - Original implementation: https://github.com/blmoistawinde/fense

    For more information, see :func:`~aac_metrics.functional.sbert.sbert`.
    """

    full_state_update = False
    higher_is_better = True
    is_differentiable = False

    min_value = -1.0
    max_value = 1.0

    def __init__(
        self,
        return_all_scores: bool = True,
        sbert_model: Union[str, SentenceTransformer] = DEFAULT_SBERT_SIM_MODEL,
        device: Union[str, torch.device, None] = "cuda_if_available",
        batch_size: int = 32,
        reset_state: bool = True,
        verbose: int = 0,
    ) -> None:
        device = _get_device(device)
        sbert_model = _load_sbert(
            sbert_model=sbert_model,
            device=device,
            reset_state=reset_state,
        )

        super().__init__()
        self._return_all_scores = return_all_scores
        self._sbert_model = sbert_model
        self._device = device
        self._batch_size = batch_size
        self._reset_state = reset_state
        self._verbose = verbose

        self._candidates = []
        self._mult_references = []

    def compute(self) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
        return sbert_sim(
            candidates=self._candidates,
            mult_references=self._mult_references,
            return_all_scores=self._return_all_scores,
            sbert_model=self._sbert_model,
            device=self._device,
            batch_size=self._batch_size,
            reset_state=self._reset_state,
            verbose=self._verbose,
        )

    def extra_repr(self) -> str:
        hparams = {"device": self._device, "batch_size": self._batch_size}
        repr_ = ", ".join(f"{k}={v}" for k, v in hparams.items())
        return repr_

    def get_output_names(self) -> tuple[str, ...]:
        return ("sbert_sim",)

    def reset(self) -> None:
        self._candidates = []
        self._mult_references = []
        return super().reset()

    def update(
        self,
        candidates: list[str],
        mult_references: list[list[str]],
    ) -> None:
        self._candidates += candidates
        self._mult_references += mult_references
