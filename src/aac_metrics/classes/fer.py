#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Union

import torch

from torch import Tensor

from aac_metrics.classes.base import AACMetric
from aac_metrics.functional.fer import (
    BERTFlatClassifier,
    fer,
    _load_echecker_and_tokenizer,
    _ERROR_NAMES,
    DEFAULT_FER_MODEL,
)
from aac_metrics.utils.globals import _get_device


pylog = logging.getLogger(__name__)


class FER(AACMetric[Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]]):
    """Return Fluency Error Rate (FER) detected by a pre-trained BERT model.

    - Paper: https://arxiv.org/abs/2110.04684
    - Original implementation: https://github.com/blmoistawinde/fense

    For more information, see :func:`~aac_metrics.functional.fer.fer`.
    """

    full_state_update = False
    higher_is_better = False
    is_differentiable = False

    min_value = -1.0
    max_value = 1.0

    def __init__(
        self,
        return_all_scores: bool = True,
        echecker: Union[str, BERTFlatClassifier] = DEFAULT_FER_MODEL,
        error_threshold: float = 0.9,
        device: Union[str, torch.device, None] = "cuda_if_available",
        batch_size: int = 32,
        reset_state: bool = True,
        return_probs: bool = False,
        verbose: int = 0,
    ) -> None:
        device = _get_device(device)
        echecker, echecker_tokenizer = _load_echecker_and_tokenizer(
            echecker=echecker,
            echecker_tokenizer=None,
            device=device,
            reset_state=reset_state,
            verbose=verbose,
        )

        super().__init__()
        self._return_all_scores = return_all_scores
        self._echecker = echecker
        self._echecker_tokenizer = echecker_tokenizer
        self._error_threshold = error_threshold
        self._device = device
        self._batch_size = batch_size
        self._reset_state = reset_state
        self._return_probs = return_probs
        self._verbose = verbose

        self._candidates = []

    def compute(self) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
        return fer(
            candidates=self._candidates,
            return_all_scores=self._return_all_scores,
            echecker=self._echecker,
            echecker_tokenizer=self._echecker_tokenizer,
            error_threshold=self._error_threshold,
            device=self._device,
            batch_size=self._batch_size,
            reset_state=self._reset_state,
            return_probs=self._return_probs,
            verbose=self._verbose,
        )

    def extra_repr(self) -> str:
        hparams = {"device": self._device, "batch_size": self._batch_size}
        repr_ = ", ".join(f"{k}={v}" for k, v in hparams.items())
        return repr_

    def get_output_names(self) -> tuple[str, ...]:
        output_names = ["fer"]
        if self._return_probs:
            output_names += [f"fer.{name}_prob" for name in _ERROR_NAMES]
        return tuple(output_names)

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
