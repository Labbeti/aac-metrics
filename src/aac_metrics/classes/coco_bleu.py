#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Union

from torch import Tensor

from aac_metrics.classes.base import Metric
from aac_metrics.functional.coco_bleu import (
    BLEU_COCO_OPTIONS,
    _coco_bleu_compute,
    _coco_bleu_update,
)


class CocoBLEU(Metric):
    """BiLingual Evaluation Understudy metric class.

    Paper: https://www.aclweb.org/anthology/P02-1040.pdf

    For more information, see :func:`~aac_metrics.functional.coco_bleu.coco_bleu`.
    """

    full_state_update = False
    higher_is_better = True
    is_differentiable = False

    min_value = 0.0
    max_value = 1.0

    def __init__(
        self,
        return_all_scores: bool = True,
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
        self._return_all_scores = return_all_scores
        self._n = n
        self._option = option
        self._verbose = verbose
        self._tokenizer = tokenizer

        self._cooked_cands = []
        self._cooked_mrefs = []

    def compute(self) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
        return _coco_bleu_compute(
            self._cooked_cands,
            self._cooked_mrefs,
            self._return_all_scores,
            self._n,
            self._option,
            self._verbose,
        )

    def extra_repr(self) -> str:
        return f"n={self._n}"

    def reset(self) -> None:
        self._cooked_cands = []
        self._cooked_mrefs = []
        return super().reset()

    def update(
        self,
        candidates: list[str],
        mult_references: list[list[str]],
    ) -> None:
        self._cooked_cands, self._cooked_mrefs = _coco_bleu_update(
            candidates,
            mult_references,
            self._n,
            self._tokenizer,
            self._cooked_cands,
            self._cooked_mrefs,
        )
