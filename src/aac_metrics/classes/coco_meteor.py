#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union

from torch import Tensor

from aac_metrics.classes.base import Metric
from aac_metrics.functional.coco_meteor import coco_meteor


class CocoMETEOR(Metric):
    """Metric for Evaluation of Translation with Explicit ORdering metric class.

    Paper: https://dl.acm.org/doi/pdf/10.5555/1626355.1626389

    For more information, see :func:`~aac_metrics.functional.coco_meteor.coco_meteor`.
    """

    full_state_update = False
    higher_is_better = True
    is_differentiable = False

    min_value = 0.0
    max_value = 1.0

    def __init__(
        self,
        return_all_scores: bool = True,
        cache_path: str = "$HOME/aac-metrics-cache",
        java_path: str = "java",
        java_max_memory: str = "2G",
        verbose: int = 0,
    ) -> None:
        super().__init__()
        self._return_all_scores = return_all_scores
        self._cache_path = cache_path
        self._java_path = java_path
        self._java_max_memory = java_max_memory
        self._verbose = verbose

        self._candidates = []
        self._mult_references = []

    def compute(self) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
        return coco_meteor(
            self._candidates,
            self._mult_references,
            self._return_all_scores,
            self._cache_path,
            self._java_path,
            self._java_max_memory,
            self._verbose,
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
