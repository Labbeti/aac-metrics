#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import math

from typing import Callable, Union

import torch

from torch import Tensor

from aac_metrics.classes.base import AACMetric
from aac_metrics.functional.vocab import vocab


pylog = logging.getLogger(__name__)


class Vocab(AACMetric[Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]]):
    """VocabStats class.

    For more information, see :func:`~aac_metrics.functional.vocab.vocab`.
    """

    full_state_update = False
    higher_is_better = True
    is_differentiable = False

    min_value = 0.0
    max_value = math.inf

    def __init__(
        self,
        return_all_scores: bool = True,
        seed: Union[None, int, torch.Generator] = 1234,
        tokenizer: Callable[[str], list[str]] = str.split,
        dtype: torch.dtype = torch.float64,
        pop_strategy: str = "max",
        verbose: int = 0,
    ) -> None:
        super().__init__()
        self._return_all_scores = return_all_scores
        self._seed = seed
        self._tokenizer = tokenizer
        self._dtype = dtype
        self._pop_strategy = pop_strategy
        self._verbose = verbose

        self._candidates = []
        self._mult_references = []

    def compute(self) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
        return vocab(
            candidates=self._candidates,
            mult_references=self._mult_references,
            return_all_scores=self._return_all_scores,
            seed=self._seed,
            tokenizer=self._tokenizer,
            dtype=self._dtype,
            pop_strategy=self._pop_strategy,
            verbose=self._verbose,
        )

    def get_output_names(self) -> tuple[str, ...]:
        return (
            "vocab",
            "vocab.mrefs_full",
            "vocab.ratio_full",
            "vocab.mrefs_avg",
            "vocab.mrefs_std",
            "vocab.ratio_avg",
        )

    def reset(self) -> None:
        self._candidates = []
        self._mult_references = []
        return super().reset()

    def update(
        self,
        candidates: list[str],
        mult_references: Union[list[list[str]], None] = None,
    ) -> None:
        self._candidates += candidates

        if mult_references is not None:
            if self._mult_references is None:
                self._mult_references = []
            else:
                self._mult_references += mult_references
        else:
            self._mult_references = None

        if self._mult_references is not None and len(self._candidates) != len(
            self._mult_references
        ):
            raise ValueError(
                f"Invalid number of sentences for {self.__class__.__name__}. (found {len(candidates)} candidates and {len(self._mult_references)} references)"
            )
