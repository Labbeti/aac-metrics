#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Union

from torch import Tensor
from torchmetrics import Metric

from aac_metrics.functional.coco_cider_d import coco_cider_d


class CocoCIDErD(Metric):
    full_state_update = False
    higher_is_better = True
    is_differentiable = False

    def __init__(
        self,
        return_all_scores: bool = False,
        n: int = 4,
        sigma: float = 6.0,
        tokenizer: Callable[[str], list[str]] = str.split,
    ) -> None:
        super().__init__()
        self.return_all_scores = return_all_scores
        self.n = n
        self.sigma = sigma
        self.tokenizer = tokenizer

        self.candidates = []
        self.mult_references = []

    def compute(self) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Tensor]]]:
        return coco_cider_d(
            self.candidates,
            self.mult_references,
            self.return_all_scores,
            self.n,
            self.sigma,
            self.tokenizer,
        )

    def reset(self) -> None:
        self.candidates = []
        self.mult_references = []
        return super().reset()

    def update(
        self,
        candidates: list[str],
        mult_references: list[list[str]],
    ) -> None:
        self.candidates += candidates
        self.mult_references += mult_references
