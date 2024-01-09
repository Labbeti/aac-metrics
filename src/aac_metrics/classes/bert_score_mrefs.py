#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Union

import torch

from torch import nn, Tensor

from aac_metrics.classes.base import AACMetric
from aac_metrics.functional.bert_score_mrefs import (
    bert_score_mrefs,
    _load_model_and_tokenizer,
    DEFAULT_BERT_SCORE_MODEL,
    REDUCTIONS,
)
from aac_metrics.utils.globals import _get_device


class BERTScoreMRefs(AACMetric):
    """BERTScore metric which supports multiple references.

    The implementation is based on the bert_score implementation of torchmetrics.

    - Paper: https://arxiv.org/pdf/1904.09675.pdf

    For more information, see :func:`~aac_metrics.functional.bert_score.bert_score_mrefs`.
    """

    full_state_update = False
    higher_is_better = True
    is_differentiable = False

    min_value = 0.0
    max_value = 1.0

    def __init__(
        self,
        return_all_scores: bool = True,
        model: Union[str, nn.Module] = DEFAULT_BERT_SCORE_MODEL,
        device: Union[str, torch.device, None] = "cuda_if_available",
        batch_size: int = 32,
        num_threads: int = 0,
        max_length: int = 64,
        reset_state: bool = True,
        idf: bool = False,
        reduction: Union[str, Callable[[Tensor, ...], Tensor]] = "max",
        filter_nan: bool = True,
        verbose: int = 0,
    ) -> None:
        if reduction not in REDUCTIONS:
            raise ValueError(
                f"Invalid argument {reduction=}. (expected one of {REDUCTIONS})"
            )

        device = _get_device(device)
        model, tokenizer = _load_model_and_tokenizer(
            model=model,
            tokenizer=None,
            device=device,
            reset_state=reset_state,
            verbose=verbose,
        )

        super().__init__()
        self._return_all_scores = return_all_scores
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._batch_size = batch_size
        self._num_threads = num_threads
        self._max_length = max_length
        self._reset_state = reset_state
        self._idf = idf
        self._reduction = reduction
        self._filter_nan = filter_nan
        self._verbose = verbose

        self._candidates = []
        self._mult_references = []

    def compute(self) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
        return bert_score_mrefs(
            candidates=self._candidates,
            mult_references=self._mult_references,
            return_all_scores=self._return_all_scores,
            model=self._model,
            tokenizer=self._tokenizer,
            device=self._device,
            batch_size=self._batch_size,
            num_threads=self._num_threads,
            max_length=self._max_length,
            reset_state=self._reset_state,
            idf=self._idf,
            reduction=self._reduction,
            filter_nan=self._filter_nan,
            verbose=self._verbose,
        )

    def extra_repr(self) -> str:
        if isinstance(self._model, str):
            model_name = self._model
        else:
            model_name = self._model.__class__.__name__

        hparams = {"model": model_name, "idf": self._idf}
        repr_ = ", ".join(f"{k}={v}" for k, v in hparams.items())
        return repr_

    def get_output_names(self) -> tuple[str, ...]:
        return (
            "bert_score.precision",
            "bert_score.recalll",
            "bert_score.f1",
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
