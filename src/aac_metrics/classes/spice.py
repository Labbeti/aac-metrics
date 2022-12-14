#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Optional, Union

from torch import Tensor

from aac_metrics.classes.base import AACMetric
from aac_metrics.functional.spice import spice


logger = logging.getLogger(__name__)


class SPICE(AACMetric):
    """Semantic Propositional Image Caption Evaluation class.

    Paper: https://arxiv.org/pdf/1607.08822.pdf

    For more information, see :func:`~aac_metrics.functional.spice.spice`.
    """

    full_state_update = False
    higher_is_better = True
    is_differentiable = False

    min_value = 0.0
    max_value = 1.0

    def __init__(
        self,
        return_all_scores: bool = True,
        cache_path: str = "$HOME/.cache",
        java_path: str = "java",
        tmp_path: str = "/tmp",
        n_threads: Optional[int] = None,
        java_max_memory: str = "8G",
        verbose: int = 0,
    ) -> None:
        super().__init__()
        self._return_all_scores = return_all_scores
        self._cache_path = cache_path
        self._java_path = java_path
        self._tmp_path = tmp_path
        self._n_threads = n_threads
        self._java_max_memory = java_max_memory
        self._verbose = verbose

        self._candidates = []
        self._mult_references = []

    def compute(self) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
        return spice(
            self._candidates,
            self._mult_references,
            self._return_all_scores,
            self._cache_path,
            self._java_path,
            self._tmp_path,
            self._n_threads,
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
