#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Union

from torch import Tensor

from aac_metrics.classes.base import Metric
from aac_metrics.functional.diversity_ratio import diversity_ratio


class DiversityRatio(Metric):
    full_state_update = False
    higher_is_better = True
    is_differentiable = False

    def __init__(
        self,
        return_all_scores: bool = True,
        tokenizer: Callable[[str], list[str]] = str.split,
    ) -> None:
        super().__init__()
        self._return_all_scores = return_all_scores
        self._tokenizer = tokenizer

        self._candidates = []
        self._mult_references = []

    def compute(self) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Tensor]]]:
        return diversity_ratio(
            self._candidates,
            self._mult_references,
            self._return_all_scores,
            self._tokenizer,
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
