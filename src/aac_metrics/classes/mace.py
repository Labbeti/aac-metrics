#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Union

import torch
from torch import Tensor
from transformers.models.auto.tokenization_auto import AutoTokenizer

from aac_metrics.classes.base import AACMetric
from aac_metrics.functional.clap_sim import CLAP, DEFAULT_CLAP_SIM_MODEL, _load_clap
from aac_metrics.functional.fer import (
    DEFAULT_FER_MODEL,
    BERTFlatClassifier,
    _load_echecker_and_tokenizer,
)
from aac_metrics.functional.mace import MACE_METHODS, MACEMethod, MACEOuts, mace
from aac_metrics.utils.globals import _get_device


class MACE(AACMetric[Union[MACEOuts, Tensor]]):
    """Multimodal Audio-Caption Evaluation class (MACE).

    MACE is a metric designed for evaluating automated audio captioning (AAC) systems.
    Unlike metrics that compare machine-generated captions solely to human references, MACE uses both audio and text to improve evaluation.
    By integrating both audio and text, it produces assessments that align better with human judgments.

    The implementation is based on the mace original implementation (original author have accepted to include their code in aac-metrics under the MIT license).
    Note: Instances of this class are not pickable.

    - Paper: https://arxiv.org/pdf/2411.00321
    - Original author: Satvik Dixit
    - Original implementation: https://github.com/satvik-dixit/mace/tree/main

    For more information, see :func:`~aac_metrics.functional.mace.mace`.
    """

    full_state_update = False
    higher_is_better = True
    is_differentiable = False

    min_value = -1.0
    max_value = 1.0

    def __init__(
        self,
        return_all_scores: bool = True,
        *,
        mace_method: MACEMethod = "text",
        penalty: float = 0.3,
        # CLAP args
        clap_model: Union[str, CLAP] = DEFAULT_CLAP_SIM_MODEL,
        seed: Optional[int] = 42,
        # FER args
        echecker: Union[str, BERTFlatClassifier] = DEFAULT_FER_MODEL,
        echecker_tokenizer: Optional[AutoTokenizer] = None,
        error_threshold: float = 0.97,
        device: Union[str, torch.device, None] = "cuda_if_available",
        batch_size: Optional[int] = 32,
        reset_state: bool = True,
        return_probs: bool = False,
        # Other args
        verbose: int = 0,
    ) -> None:
        if mace_method not in MACE_METHODS:
            msg = f"Invalid argument {mace_method=}. (expected one of {MACE_METHODS})"
            raise ValueError(msg)

        device = _get_device(device)
        clap_model = _load_clap(
            clap_model=clap_model,
            device=device,
            reset_state=reset_state,
        )
        echecker, echecker_tokenizer = _load_echecker_and_tokenizer(
            echecker=echecker,
            echecker_tokenizer=echecker_tokenizer,
            device=device,
            reset_state=reset_state,
            verbose=verbose,
        )

        super().__init__()
        self._return_all_scores = return_all_scores
        self._mace_method: MACEMethod = mace_method
        self._penalty = penalty
        self._clap_model = clap_model
        self._seed = seed
        self._echecker = echecker
        self._echecker_tokenizer = echecker_tokenizer
        self._error_threshold = error_threshold
        self._device = device
        self._batch_size = batch_size
        self._reset_state = reset_state
        self._return_probs = return_probs
        self._verbose = verbose

        self._candidates = []
        self._mult_references = []
        self._audio_paths = []

    def compute(self) -> Union[MACEOuts, Tensor]:
        return mace(
            candidates=self._candidates,
            mult_references=self._mult_references,
            audio_paths=self._audio_paths,
            return_all_scores=self._return_all_scores,
            mace_method=self._mace_method,
            clap_model=self._clap_model,
            seed=self._seed,
            echecker=self._echecker,
            echecker_tokenizer=self._echecker_tokenizer,
            error_threshold=self._error_threshold,
            device=self._device,
            batch_size=self._batch_size,
            reset_state=self._reset_state,
            return_probs=self._return_probs,
            penalty=self._penalty,
            verbose=self._verbose,
        )

    def extra_repr(self) -> str:
        if isinstance(self._clap_model, str):
            model_name = self._clap_model
        else:
            model_name = self._clap_model.__class__.__name__

        hparams = {"clap_model": model_name}
        repr_ = ", ".join(f"{k}={v}" for k, v in hparams.items())
        return repr_

    def get_output_names(self) -> tuple[str, ...]:
        return ("mace", "fer", "clap_sim")

    def reset(self) -> None:
        self._candidates = []
        self._mult_references = []
        self._audio_paths = []
        return super().reset()

    def update(
        self,
        candidates: list[str],
        mult_references: Optional[list[list[str]]] = None,
        audio_paths: Optional[list[str]] = None,
    ) -> None:
        self._candidates += candidates

        if self._mace_method == "audio":
            if mult_references is None:
                msg = f"Invalid argument {mult_references=} with {self._mace_method=}."
                raise ValueError(msg)

            self._mult_references += mult_references

        elif self._mace_method == "text":
            if audio_paths is None:
                msg = f"Invalid argument {audio_paths=} with {self._mace_method=}."
                raise ValueError(msg)

            self._audio_paths += audio_paths

        elif self._mace_method == "combined":
            if mult_references is None:
                msg = f"Invalid argument {mult_references=} with {self._mace_method=}."
                raise ValueError(msg)
            if audio_paths is None:
                msg = f"Invalid argument {audio_paths=} with {self._mace_method=}."
                raise ValueError(msg)

            self._mult_references += mult_references
            self._audio_paths += audio_paths

        else:
            msg = (
                f"Invalid value {self._mace_method=}. (expected one of {MACE_METHODS})"
            )
            raise ValueError(msg)

    def __getstate__(self) -> bytes:
        raise RuntimeError(f"{self.__class__.__name__} is not pickable.")
