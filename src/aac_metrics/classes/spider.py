#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from pathlib import Path
from typing import Iterable, Optional, Union

from torch import Tensor

from aac_metrics.classes.base import AACMetric
from aac_metrics.functional.spider import spider


pylog = logging.getLogger(__name__)


class SPIDEr(AACMetric[Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]]):
    """SPIDEr class.

    - Paper: https://arxiv.org/pdf/1612.00370.pdf

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
        # CIDErD args
        n: int = 4,
        sigma: float = 6.0,
        # SPICE args
        cache_path: Union[str, Path, None] = None,
        java_path: Union[str, Path, None] = None,
        tmp_path: Union[str, Path, None] = None,
        n_threads: Optional[int] = None,
        java_max_memory: str = "8G",
        timeout: Union[None, int, Iterable[int]] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__()
        self._return_all_scores = return_all_scores
        self._n = n
        self._sigma = sigma
        self._cache_path = cache_path
        self._java_path = java_path
        self._tmp_path = tmp_path
        self._n_threads = n_threads
        self._java_max_memory = java_max_memory
        self._timeout = timeout
        self._verbose = verbose

        self._candidates = []
        self._mult_references = []

    def compute(self) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
        return spider(
            candidates=self._candidates,
            mult_references=self._mult_references,
            return_all_scores=self._return_all_scores,
            # CIDEr args
            n=self._n,
            sigma=self._sigma,
            # SPICE args
            cache_path=self._cache_path,
            java_path=self._java_path,
            tmp_path=self._tmp_path,
            n_threads=self._n_threads,
            java_max_memory=self._java_max_memory,
            timeout=self._timeout,
            verbose=self._verbose,
        )

    def extra_repr(self) -> str:
        hparams = {
            "n": self._n,
            "sigma": self._sigma,
            "java_max_memory": self._java_max_memory,
        }
        repr_ = ", ".join(f"{k}={v}" for k, v in hparams.items())
        return repr_

    def get_output_names(self) -> tuple[str, ...]:
        return ("cider_d", "spice", "spider")

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
