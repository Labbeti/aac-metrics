#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Callable, Iterable, Union

from torch import Tensor

from aac_metrics.functional.evaluate import evaluate, _get_metrics_set
from aac_metrics.classes.base import Metric


logger = logging.getLogger(__name__)


class Evaluate(Metric, list[Metric]):
    full_state_update = False
    higher_is_better = True
    is_differentiable = False

    def __init__(
        self,
        use_ptb_tokenizer: bool = True,
        java_path: str = "java",
        tmp_path: str = "/tmp",
        cache_path: str = "$HOME/aac-metrics-cache",
        verbose: int = 0,
        metrics: Union[str, Iterable[Callable]] = "all",
    ) -> None:
        if isinstance(metrics, str):
            metrics = _get_metrics_set(
                metrics,
                java_path=java_path,
                tmp_path=tmp_path,
                cache_path=cache_path,
                verbose=verbose,
            )

        Metric.__init__(self)
        list.__init__(self, metrics)
        self._use_ptb_tokenizer = use_ptb_tokenizer
        self._java_path = java_path
        self._tmp_path = tmp_path
        self._cache_path = cache_path
        self._verbose = verbose

        self._candidates = []
        self._mult_references = []

    def compute(self) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        return evaluate(
            self._candidates,
            self._mult_references,
            self._use_ptb_tokenizer,
            self,
            java_path=self._java_path,
            tmp_path=self._tmp_path,
            cache_path=self._cache_path,
            verbose=self._verbose,
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