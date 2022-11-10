#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Union

import torch

from sentence_transformers import SentenceTransformer
from torch import Tensor
from transformers.models.auto.tokenization_auto import AutoTokenizer

from aac_metrics.classes.base import Metric
from aac_metrics.functional.fense import fense, _load_pretrain_echecker


logger = logging.getLogger(__name__)


class FENSE(Metric):
    """Fluency ENhanced Sentence-bert Evaluation (FENSE)

    Paper: https://arxiv.org/abs/2110.04684
    Original implementation: https://github.com/blmoistawinde/fense

    For more information, see :func:`~aac_metrics.functional.fense.fense`.
    """

    full_state_update = False
    higher_is_better = True
    is_differentiable = False

    min_value = 0.0
    max_value = 1.0

    def __init__(
        self,
        return_all_scores: bool = True,
        sbert_model: str = "paraphrase-TinyBERT-L6-v2",
        echecker: Union[None, str] = "echecker_clotho_audiocaps_base",
        error_threshold: float = 0.9,
        penalty: float = 0.9,
        agg_score: str = "mean",
        device: Union[torch.device, str, None] = "cpu",
        batch_size: int = 32,
        verbose: int = 0,
    ) -> None:
        super().__init__()
        self._return_all_scores = return_all_scores
        self._error_threshold = error_threshold
        self._penalty = penalty
        self._agg_score = agg_score
        self._device = device
        self._batch_size = batch_size
        self._verbose = verbose

        self._sbert_model = SentenceTransformer(sbert_model, device=device)  # type: ignore
        if echecker in (None, "none"):
            self._echecker = None
            self._echecker_tokenizer = None
        else:
            self._echecker = _load_pretrain_echecker(echecker, device)
            self._echecker_tokenizer = AutoTokenizer.from_pretrained(
                self._echecker.model_type
            )

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
            self._agg_score,
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
