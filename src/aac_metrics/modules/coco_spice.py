#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from typing import Optional, Union

from torch import Tensor
from torchmetrics import Metric

from aac_metrics.functional.coco_spice import coco_spice


logger = logging.getLogger(__name__)


class CocoSPICE(Metric):
    full_state_update = False
    higher_is_better = True
    is_differentiable = False

    def __init__(
        self,
        return_all_scores: bool = False,
        java_path: str = "java",
        tmp_path: str = "/tmp",
        cache_path: str = "$HOME/aac-metrics-cache",
        n_threads: Optional[int] = None,
        java_max_memory: str = "8G",
        verbose: int = 0,
    ) -> None:
        super().__init__()
        self.return_all_scores = return_all_scores
        self.java_path = java_path
        self.tmp_path = tmp_path
        self.cache_path = cache_path
        self.n_threads = n_threads
        self.java_max_memory = java_max_memory
        self.verbose = verbose

        self.candidates = []
        self.mult_references = []

    def compute(self) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Tensor]]]:
        return coco_spice(
            self.candidates,
            self.mult_references,
            self.return_all_scores,
            self.java_path,
            self.tmp_path,
            self.cache_path,
            self.n_threads,
            self.java_max_memory,
            self.verbose,
        )

    def reset(self) -> None:
        self.candidates = []
        self.mult_references = []
        return super().reset()

    def update(
        self,
        candidates: list[str],
        mult_references: list[list[str]],
    ) -> None:
        self.candidates += candidates
        self.mult_references += mult_references
