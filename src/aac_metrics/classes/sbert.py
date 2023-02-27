#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Union

import torch

from torch import Tensor

from aac_metrics.classes.base import AACMetric
from aac_metrics.functional.sbert import sbert, _load_sbert


logger = logging.getLogger(__name__)


class SBERT(AACMetric):
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
        sbert_model: str = "paraphrase-TinyBERT-L6-v2",
        device: Union[str, torch.device, None] = "auto",
        batch_size: int = 32,
        reset_state: bool = True,
        verbose: int = 0,
    ) -> None:
        sbert_model = _load_sbert(sbert_model, device, reset_state, verbose)  # type: ignore

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
        return sbert(
            self._candidates,
            self._mult_references,
            self._return_all_scores,
            self._sbert_model,
            self._device,
            self._batch_size,
            self._reset_state,
            self._verbose,
        )

    def extra_repr(self) -> str:
        return f"device={self._device}, batch_size={self._batch_size}"

    def get_output_names(self) -> tuple[str, ...]:
        return ("sbert.sim",)

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
