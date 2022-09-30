#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Iterable, Union

from torch import Tensor

from aac_metrics.classes.base import Metric
from aac_metrics.functional.evaluate import custom_evaluate, _get_metrics_list


logger = logging.getLogger(__name__)


class CustomEvaluate(Metric, list[Metric]):
    """Evaluate candidates with multiple references with custom metrics.

    For more information, see :func:`~aac_metrics.functional.evaluate.custom_evaluate`.
    """

    full_state_update = False
    higher_is_better = None
    is_differentiable = False

    def __init__(
        self,
        use_ptb_tokenizer: bool = True,
        cache_path: str = "$HOME/aac-metrics-cache",
        java_path: str = "java",
        tmp_path: str = "/tmp",
        verbose: int = 0,
        metrics: Union[str, Iterable[Metric]] = "all",
    ) -> None:
        if isinstance(metrics, str):
            metrics = _get_metrics_list(
                metrics,
                cache_path=cache_path,
                java_path=java_path,
                tmp_path=tmp_path,
                verbose=verbose,
            )

        Metric.__init__(self)
        list.__init__(self, metrics)
        self._use_ptb_tokenizer = use_ptb_tokenizer
        self._cache_path = cache_path
        self._java_path = java_path
        self._tmp_path = tmp_path
        self._verbose = verbose

        self._candidates = []
        self._mult_references = []

    def compute(self) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        return custom_evaluate(
            self._candidates,
            self._mult_references,
            self._use_ptb_tokenizer,
            self,
            cache_path=self._cache_path,
            java_path=self._java_path,
            tmp_path=self._tmp_path,
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


class AACEvaluate(CustomEvaluate):
    """Evaluate candidates with multiple references with all Audio Captioning metrics.

    For more information, see :func:`~aac_metrics.functional.evaluate.aac_evaluate`.
    """

    def __init__(
        self,
        use_ptb_tokenizer: bool = True,
        cache_path: str = "$HOME/aac-metrics-cache",
        java_path: str = "java",
        tmp_path: str = "/tmp",
        verbose: int = 0,
    ) -> None:
        super().__init__(
            use_ptb_tokenizer,
            cache_path,
            java_path,
            tmp_path,
            verbose,
            "aac",
        )
