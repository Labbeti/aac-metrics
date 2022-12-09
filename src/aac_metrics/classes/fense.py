#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Union

import torch

from torch import Tensor

from aac_metrics.classes.base import AACMetric
from aac_metrics.functional.fense import fense, _load_models_and_tokenizer


logger = logging.getLogger(__name__)


class FENSE(AACMetric):
    """Fluency ENhanced Sentence-bert Evaluation (FENSE)

    Paper: https://arxiv.org/abs/2110.04684
    Original implementation: https://github.com/blmoistawinde/fense

    For more information, see :func:`~aac_metrics.functional.fense.fense`.
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
        echecker: Union[None, str] = "echecker_clotho_audiocaps_base",
        error_threshold: float = 0.9,
        penalty: float = 0.9,
        device: Union[str, torch.device, None] = "cpu",
        batch_size: int = 32,
        verbose: int = 0,
    ) -> None:
        sbert_model, echecker, echecker_tokenizer = _load_models_and_tokenizer(sbert_model, echecker, None, device)  # type: ignore

        super().__init__()
        self._return_all_scores = return_all_scores
        self._sbert_model = sbert_model
        self._echecker = echecker
        self._echecker_tokenizer = echecker_tokenizer
        self._error_threshold = error_threshold
        self._penalty = penalty
        self._device = device
        self._batch_size = batch_size
        self._verbose = verbose

        self._candidates = []
        self._mult_references = []

    def compute(self) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
        return fense(
            self._candidates,
            self._mult_references,
            self._return_all_scores,
            self._sbert_model,
            self._echecker,
            self._echecker_tokenizer,
            self._error_threshold,
            self._penalty,
            self._device,
            self._batch_size,
            self._verbose,
        )

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
