#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Union

from torch import Tensor

from aac_metrics.functional.coco_bleu import coco_bleu, BLEU_COCO_OPTIONS
from aac_metrics.modules.base import Metric


class CocoBLEU(Metric):
    full_state_update = False
    higher_is_better = True
    is_differentiable = False

    def __init__(
        self,
        return_all_scores: bool = False,
        n: int = 4,
        option: str = "closest",
        verbose: int = 0,
        tokenizer: Callable[[str], list[str]] = str.split,
    ) -> None:
        if option not in BLEU_COCO_OPTIONS:
            raise ValueError(
                f"Invalid option {option=}. (expected one of {BLEU_COCO_OPTIONS})"
            )

        super().__init__()
        self.return_all_scores = return_all_scores
        self.n = n
        self.option = option
        self.verbose = verbose
        self.tokenizer = tokenizer

        self.candidates = []
        self.mult_references = []

    def compute(self) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Tensor]]]:
        return coco_bleu(
            self.candidates,
            self.mult_references,
            self.return_all_scores,
            self.n,
            self.option,
            self.verbose,
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
        if len(candidates) != len(mult_references):
            raise ValueError(
                f"Invalid number of candidates and references. (found {len(candidates)=} != {len(mult_references)=})"
            )

        self.candidates += candidates
        self.mult_references += mult_references
