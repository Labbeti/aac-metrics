#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Union

import torch

from torch import Tensor

from aac_metrics.classes.base import AACMetric
from aac_metrics.functional.fense import fense, _load_models_and_tokenizer
from aac_metrics.functional.fluerr import ERROR_NAMES


pylog = logging.getLogger(__name__)


class FENSE(AACMetric):
    """Fluency ENhanced Sentence-bert Evaluation (FENSE)

    - Paper: https://arxiv.org/abs/2110.04684
    - Original implementation: https://github.com/blmoistawinde/fense

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
        echecker: str = "echecker_clotho_audiocaps_base",
        error_threshold: float = 0.9,
        device: Union[str, torch.device, None] = "auto",
        batch_size: int = 32,
        reset_state: bool = True,
        return_probs: bool = True,
        penalty: float = 0.9,
        verbose: int = 0,
    ) -> None:
        sbert_model, echecker, echecker_tokenizer = _load_models_and_tokenizer(sbert_model, echecker, None, device, reset_state, verbose)  # type: ignore

        super().__init__()
        self._return_all_scores = return_all_scores
        self._sbert_model = sbert_model
        self._echecker = echecker
        self._echecker_tokenizer = echecker_tokenizer
        self._error_threshold = error_threshold
        self._device = device
        self._batch_size = batch_size
        self._reset_state = reset_state
        self._return_probs = return_probs
        self._penalty = penalty
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
            self._device,
            self._batch_size,
            self._reset_state,
            self._return_probs,
            self._penalty,
            self._verbose,
        )

    def extra_repr(self) -> str:
        return f"error_threshold={self._error_threshold}, penalty={self._penalty}, device={self._device}, batch_size={self._batch_size}"

    def get_output_names(self) -> tuple[str, ...]:
        return ("sbert_sim", "fluerr", "fense") + tuple(
            f"fluerr.{name}_prob" for name in ERROR_NAMES
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
