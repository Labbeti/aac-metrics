#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Union

import torch

from torch import Tensor

from aac_metrics.classes.base import AACMetric
from aac_metrics.functional.fluency_error import (
    fluency_error,
    _load_echecker_and_tokenizer,
    ERROR_NAMES,
)


logger = logging.getLogger(__name__)


class FluencyError(AACMetric):
    """Return fluency error detected by a pre-trained BERT model.

    - Paper: https://arxiv.org/abs/2110.04684
    - Original implementation: https://github.com/blmoistawinde/fense

    For more information, see :func:`~aac_metrics.functional.fluency_error.fluency_error`.
    """

    full_state_update = False
    higher_is_better = False
    is_differentiable = False

    min_value = -1.0
    max_value = 1.0

    def __init__(
        self,
        return_all_scores: bool = True,
        echecker: str = "echecker_clotho_audiocaps_base",
        error_threshold: float = 0.9,
        device: Union[str, torch.device, None] = "auto",
        batch_size: int = 32,
        reset_state: bool = True,
        verbose: int = 0,
    ) -> None:
        echecker, echecker_tokenizer = _load_echecker_and_tokenizer(echecker, None, device, reset_state, verbose)  # type: ignore

        super().__init__()
        self._return_all_scores = return_all_scores
        self._echecker = echecker
        self._echecker_tokenizer = echecker_tokenizer
        self._error_threshold = error_threshold
        self._device = device
        self._batch_size = batch_size
        self._reset_state = reset_state
        self._verbose = verbose

        self._candidates = []

    def compute(self) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
        return fluency_error(
            self._candidates,
            self._return_all_scores,
            self._echecker,
            self._echecker_tokenizer,
            self._error_threshold,
            self._device,
            self._batch_size,
            self._reset_state,
            self._verbose,
        )

    def extra_repr(self) -> str:
        return f"device={self._device}, batch_size={self._batch_size}"

    def get_output_names(self) -> tuple[str, ...]:
        return ("fluency_error",) + tuple(f"fluerr.{name}_prob" for name in ERROR_NAMES)

    def reset(self) -> None:
        self._candidates = []
        return super().reset()

    def update(
        self,
        candidates: list[str],
        *args,
        **kwargs,
    ) -> None:
        self._candidates += candidates
