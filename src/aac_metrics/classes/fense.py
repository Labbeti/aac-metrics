#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Union

import torch

from sentence_transformers import SentenceTransformer
from torch import Tensor

from aac_metrics.classes.base import AACMetric
from aac_metrics.functional.fense import fense, _load_models_and_tokenizer
from aac_metrics.functional.fer import (
    BERTFlatClassifier,
    _ERROR_NAMES,
    DEFAULT_FER_MODEL,
)
from aac_metrics.functional.sbert_sim import DEFAULT_SBERT_SIM_MODEL


pylog = logging.getLogger(__name__)


class FENSE(AACMetric[Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]]):
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
        sbert_model: Union[str, SentenceTransformer] = DEFAULT_SBERT_SIM_MODEL,
        echecker: Union[str, BERTFlatClassifier] = DEFAULT_FER_MODEL,
        error_threshold: float = 0.9,
        device: Union[str, torch.device, None] = "cuda_if_available",
        batch_size: int = 32,
        reset_state: bool = True,
        return_probs: bool = False,
        penalty: float = 0.9,
        verbose: int = 0,
    ) -> None:
        sbert_model, echecker, echecker_tokenizer = _load_models_and_tokenizer(
            sbert_model=sbert_model,
            echecker=echecker,
            echecker_tokenizer=None,
            device=device,
            reset_state=reset_state,
            verbose=verbose,
        )

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
            candidates=self._candidates,
            mult_references=self._mult_references,
            return_all_scores=self._return_all_scores,
            sbert_model=self._sbert_model,
            echecker=self._echecker,
            echecker_tokenizer=self._echecker_tokenizer,
            error_threshold=self._error_threshold,
            device=self._device,
            batch_size=self._batch_size,
            reset_state=self._reset_state,
            return_probs=self._return_probs,
            penalty=self._penalty,
            verbose=self._verbose,
        )

    def extra_repr(self) -> str:
        hparams = {
            "error_threshold": self._error_threshold,
            "penalty": self._penalty,
            "device": self._device,
            "batch_size": self._batch_size,
        }
        repr_ = ", ".join(f"{k}={v}" for k, v in hparams.items())
        return repr_

    def get_output_names(self) -> tuple[str, ...]:
        output_names = ["sbert_sim", "fer", "fense"]
        if self._return_probs:
            output_names += [f"fer.{name}_prob" for name in _ERROR_NAMES]
        return tuple(output_names)

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
