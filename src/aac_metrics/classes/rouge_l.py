#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Union

from torch import Tensor

from aac_metrics.classes.base import AACMetric
from aac_metrics.functional.rouge_l import (
    _rouge_l_compute,
    _rouge_l_update,
)


class ROUGEL(AACMetric[Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]]):
    """Recall-Oriented Understudy for Gisting Evaluation class.

    - Paper: https://aclanthology.org/W04-1013.pdf

    For more information, see :func:`~aac_metrics.functional.rouge_l.rouge_l`.
    """

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
        self._return_all_scores = return_all_scores
        self._beta = beta
        self._tokenizer = tokenizer

        self._rouge_l_scores = []

    def compute(self) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
        return _rouge_l_compute(
            rouge_l_scs=self._rouge_l_scores,
            return_all_scores=self._return_all_scores,
        )

    def extra_repr(self) -> str:
        hparams = {"beta": self._beta}
        repr_ = ", ".join(f"{k}={v}" for k, v in hparams.items())
        return repr_

    def get_output_names(self) -> tuple[str, ...]:
        return ("rouge_l",)

    def reset(self) -> None:
        self._rouge_l_scores = []
        return super().reset()

    def update(
        self,
        candidates: list[str],
        mult_references: list[list[str]],
    ) -> None:
        self._rouge_l_scores = _rouge_l_update(
            candidates=candidates,
            mult_references=mult_references,
            beta=self._beta,
            tokenizer=self._tokenizer,
            prev_rouge_l_scores=self._rouge_l_scores,
        )
