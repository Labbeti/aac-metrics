#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Iterable, Optional, Union

from torch import Tensor

from aac_metrics.classes.base import AACMetric
from aac_metrics.functional.meteor import meteor


class METEOR(AACMetric[Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]]):
    """Metric for Evaluation of Translation with Explicit ORdering metric class.

    - Paper: https://dl.acm.org/doi/pdf/10.5555/1626355.1626389
    - Documentation: https://www.cs.cmu.edu/~alavie/METEOR/README.html

    For more information, see :func:`~aac_metrics.functional.meteor.meteor`.
    """

    full_state_update = False
    higher_is_better = True
    is_differentiable = False

    min_value = 0.0
    max_value = 1.0

    def __init__(
        self,
        return_all_scores: bool = True,
        cache_path: Union[str, Path, None] = None,
        java_path: Union[str, Path, None] = None,
        java_max_memory: str = "2G",
        language: str = "en",
        use_shell: Optional[bool] = None,
        params: Optional[Iterable[float]] = None,
        weights: Optional[Iterable[float]] = None,
        verbose: int = 0,
    ) -> None:
        super().__init__()
        self._return_all_scores = return_all_scores
        self._cache_path = cache_path
        self._java_path = java_path
        self._java_max_memory = java_max_memory
        self._language = language
        self._use_shell = use_shell
        self._params = params
        self._weights = weights
        self._verbose = verbose

        self._candidates = []
        self._mult_references = []

    def compute(self) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
        return meteor(
            candidates=self._candidates,
            mult_references=self._mult_references,
            return_all_scores=self._return_all_scores,
            cache_path=self._cache_path,
            java_path=self._java_path,
            java_max_memory=self._java_max_memory,
            language=self._language,
            use_shell=self._use_shell,
            params=self._params,
            weights=self._weights,
            verbose=self._verbose,
        )

    def extra_repr(self) -> str:
        hparams = {"java_max_memory": self._java_max_memory, "language": self._language}
        repr_ = ", ".join(f"{k}={v}" for k, v in hparams.items())
        return repr_

    def get_output_names(self) -> tuple[str, ...]:
        return ("meteor",)

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
