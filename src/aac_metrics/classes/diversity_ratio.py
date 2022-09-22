#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

from typing import Callable, Union

from torch import Tensor

from aac_metrics.classes.base import Metric
from aac_metrics.functional.diversity_ratio import (
    _diversity_ratio_compute,
    _diversity_ratio_update,
)


class DiversityRatio(Metric):
    full_state_update = False
    higher_is_better = True
    is_differentiable = False

    min_value = 0.0
    max_value = math.inf
    is_linear = True

    def __init__(
        self,
        return_all_scores: bool = True,
        tokenizer: Callable[[str], list[str]] = str.split,
    ) -> None:
        super().__init__()
        self._return_all_scores = return_all_scores
        self._tokenizer = tokenizer

        self._tok_cands = []
        self._tok_mrefs = []

    def compute(self) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Tensor]]]:
        return _diversity_ratio_compute(
            self._tok_cands,
            self._tok_mrefs,
            self._return_all_scores,
        )

    def reset(self) -> None:
        self._tok_cands = []
        self._tok_mrefs = []
        return super().reset()

    def update(
        self,
        candidates: list[str],
        mult_references: list[list[str]],
    ) -> None:
        self._tok_cands, self._tok_mrefs = _diversity_ratio_update(
            candidates,
            mult_references,
            self._tokenizer,
            self._tok_cands,
            self._tok_mrefs,
        )
