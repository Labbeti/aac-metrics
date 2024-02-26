#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Callable, Union

from torch import Tensor

from aac_metrics.classes.base import AACMetric
from aac_metrics.functional.cider_d import (
    _cider_d_compute,
    _cider_d_update,
)


class CIDErD(AACMetric[Union[tuple[dict[str, Tensor], dict[str, Any]], Tensor]]):
    """Consensus-based Image Description Evaluation metric class.

    - Paper: https://arxiv.org/pdf/1411.5726.pdf

    For more information, see :func:`~aac_metrics.functional.cider_d.cider_d`.
    """

    full_state_update = False
    higher_is_better = True
    is_differentiable = False

    min_value = 0.0
    max_value = 10.0

    def __init__(
        self,
        return_all_scores: bool = True,
        n: int = 4,
        sigma: float = 6.0,
        tokenizer: Callable[[str], list[str]] = str.split,
        return_tfidf: bool = False,
        scale: float = 10.0,
    ) -> None:
        super().__init__()
        self._return_all_scores = return_all_scores
        self._n = n
        self._sigma = sigma
        self._tokenizer = tokenizer
        self._return_tfidf = return_tfidf
        self._scale = scale

        self._cooked_cands = []
        self._cooked_mrefs = []

    def compute(self) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
        return _cider_d_compute(
            cooked_cands=self._cooked_cands,
            cooked_mrefs=self._cooked_mrefs,
            return_all_scores=self._return_all_scores,
            n=self._n,
            sigma=self._sigma,
            return_tfidf=self._return_tfidf,
            scale=self._scale,
        )

    def extra_repr(self) -> str:
        hparams = {"n": self._n, "sigma": self._sigma}
        repr_ = ", ".join(f"{k}={v}" for k, v in hparams.items())
        return repr_

    def get_output_names(self) -> tuple[str, ...]:
        return ("cider_d",)

    def reset(self) -> None:
        self._cooked_cands = []
        self._cooked_mrefs = []
        return super().reset()

    def update(
        self,
        candidates: list[str],
        mult_references: list[list[str]],
    ) -> None:
        self._cooked_cands, self._cooked_mrefs = _cider_d_update(
            candidates=candidates,
            mult_references=mult_references,
            n=self._n,
            tokenizer=self._tokenizer,
            prev_cooked_cands=self._cooked_cands,
            prev_cooked_mrefs=self._cooked_mrefs,
        )
