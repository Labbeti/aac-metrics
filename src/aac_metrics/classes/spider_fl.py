#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

from pathlib import Path
from typing import Iterable, Optional, Union

import torch

from torch import Tensor
from transformers.models.auto.tokenization_auto import AutoTokenizer

from aac_metrics.classes.base import AACMetric
from aac_metrics.functional.fer import (
    BERTFlatClassifier,
    _load_echecker_and_tokenizer,
    _ERROR_NAMES,
    DEFAULT_FER_MODEL,
)
from aac_metrics.functional.spider_fl import spider_fl
from aac_metrics.utils.globals import _get_device


pylog = logging.getLogger(__name__)


class SPIDErFL(AACMetric[Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]]):
    """SPIDErFL class.

    For more information, see :func:`~aac_metrics.functional.spider_fl.spider_fl`.
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
        # FluencyError args
        echecker: Union[str, BERTFlatClassifier] = DEFAULT_FER_MODEL,
        echecker_tokenizer: Optional[AutoTokenizer] = None,
        error_threshold: float = 0.9,
        device: Union[str, torch.device, None] = "cuda_if_available",
        batch_size: int = 32,
        reset_state: bool = True,
        return_probs: bool = True,
        # Other args
        penalty: float = 0.9,
        verbose: int = 0,
    ) -> None:
        device = _get_device(device)
        echecker, echecker_tokenizer = _load_echecker_and_tokenizer(
            echecker=echecker,
            echecker_tokenizer=echecker_tokenizer,
            device=device,
            reset_state=reset_state,
            verbose=verbose,
        )

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
        self._echecker = echecker
        self._echecker_tokenizer = echecker_tokenizer
        self._error_threshold = error_threshold
        self._device = device
        self._batch_size = batch_size
        self._reset_state = reset_state
        self._return_probs = return_probs
        self._penalty = penalty
        self._verbose = verbose

        self._candidates = []
        self._mult_references = []

    def compute(self) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
        return spider_fl(
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
            # FluencyError args
            echecker=self._echecker,
            echecker_tokenizer=self._echecker_tokenizer,
            error_threshold=self._error_threshold,
            device=self._device,
            batch_size=self._batch_size,
            reset_state=self._reset_state,
            return_probs=self._return_probs,
            # Other args
            penalty=self._penalty,
            verbose=self._verbose,
        )

    def extra_repr(self) -> str:
        hparams = {
            "n": self._n,
            "sigma": self._sigma,
            "java_max_memory": self._java_max_memory,
            "device": self._device,
            "batch_size": self._batch_size,
        }
        extra = ", ".join(f"{k}={v}" for k, v in hparams.items())
        return extra

    def get_output_names(self) -> tuple[str, ...]:
        output_names = ["cider_d", "spice", "spider", "spider_fl", "fer"]
        if self._return_probs:
            output_names += [f"fer.{name}_prob" for name in _ERROR_NAMES]
        return tuple(output_names)

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
