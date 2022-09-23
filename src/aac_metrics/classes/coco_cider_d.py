#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Union

from torch import Tensor

from aac_metrics.classes.base import Metric
from aac_metrics.functional.coco_cider_d import (
    _coco_cider_d_compute,
    _coco_cider_d_update,
)


class CocoCIDErD(Metric):
    full_state_update = False
    higher_is_better = True
    is_differentiable = False

    min_value = 0.0
    max_value = 10.0
    is_linear = False

    def __init__(
        self,
        return_all_scores: bool = True,
        n: int = 4,
        sigma: float = 6.0,
        tokenizer: Callable[[str], list[str]] = str.split,
    ) -> None:
        super().__init__()
        self._return_all_scores = return_all_scores
        self._n = n
        self._sigma = sigma
        self._tokenizer = tokenizer

        self._cooked_cands = []
        self._cooked_mrefs = []

    def compute(self) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Tensor]]]:
        return _coco_cider_d_compute(
            self._cooked_cands,
            self._cooked_mrefs,
            self._return_all_scores,
            self._n,
            self._sigma,
        )

    def reset(self) -> None:
        self._cooked_cands = []
        self._cooked_mrefs = []
        return super().reset()

    def update(
        self,
        candidates: list[str],
        mult_references: list[list[str]],
    ) -> None:
        self._cooked_cands, self._cooked_mrefs = _coco_cider_d_update(
            candidates,
            mult_references,
            self._n,
            self._tokenizer,
            self._cooked_cands,
            self._cooked_mrefs,
        )
