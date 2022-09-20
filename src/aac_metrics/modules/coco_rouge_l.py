#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Union

from torch import Tensor
from torchmetrics import Metric

from aac_metrics.functional.coco_rouge_l import coco_rouge_l


class CocoRougeL(Metric):
    full_state_update = False
    higher_is_better = True
    is_differentiable = False

    min_value = 0.0
    max_value = 1.0

    def __init__(
        self,
        return_all_scores: bool = True,
        beta: float = 1.2,
        tokenizer: Callable[[str], list[str]] = str.split,
    ) -> None:
        super().__init__()
        self.return_all_scores = return_all_scores
        self.beta = beta
        self.tokenizer = tokenizer

        self.candidates = []
        self.mult_references = []

    def compute(self) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Tensor]]]:
        return coco_rouge_l(
            self.candidates,
            self.mult_references,
            self.return_all_scores,
            self.beta,
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
