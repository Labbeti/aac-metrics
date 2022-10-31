#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Optional, Union

from torch import Tensor

from aac_metrics.classes.base import Metric
from aac_metrics.functional.spider_max import spider_max


logger = logging.getLogger(__name__)


class SPIDErMax(Metric):
    """SPIDEr-max class.

    Paper: https://hal.archives-ouvertes.fr/hal-03810396/file/Labbe_DCASE2022.pdf

    For more information, see :func:`~aac_metrics.functional.spider.spider`.
    """

    full_state_update = False
    higher_is_better = True
    is_differentiable = False

    min_value = 0.0
    max_value = 5.5

    def __init__(
        self,
        return_all_scores: bool = True,
        return_all_cands_scores: bool = False,
        # CIDEr args
        n: int = 4,
        sigma: float = 6.0,
        # SPICE args
        cache_path: str = "$HOME/aac-metrics-cache",
        java_path: str = "java",
        tmp_path: str = "/tmp",
        n_threads: Optional[int] = None,
        java_max_memory: str = "8G",
        verbose: int = 0,
    ) -> None:
        super().__init__()
        self._return_all_scores = return_all_scores
        self._return_all_cands_scores = return_all_cands_scores
        self._n = n
        self._sigma = sigma
        self._cache_path = cache_path
        self._java_path = java_path
        self._tmp_path = tmp_path
        self._n_threads = n_threads
        self._java_max_memory = java_max_memory
        self._verbose = verbose

        self._mult_candidates = []
        self._mult_references = []

    def compute(self) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
        return spider_max(
            self._mult_candidates,
            self._mult_references,
            self._return_all_scores,
            self._return_all_cands_scores,
            n=self._n,
            sigma=self._sigma,
            cache_path=self._cache_path,
            java_path=self._java_path,
            tmp_path=self._tmp_path,
            n_threads=self._n_threads,
            java_max_memory=self._java_max_memory,
            verbose=self._verbose,
        )

    def reset(self) -> None:
        self._mult_candidates = []
        self._mult_references = []
        return super().reset()

    def update(
        self,
        mult_candidates: list[list[str]],
        mult_references: list[list[str]],
    ) -> None:
        self._mult_candidates += mult_candidates
        self._mult_references += mult_references
