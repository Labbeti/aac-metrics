#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Optional, Union

import torch
from torch import Tensor

from aac_metrics.classes.base import AACMetric
from aac_metrics.functional.clap_sim import (
    CLAP,
    CLAP_METHODS,
    DEFAULT_CLAP_SIM_MODEL,
    CLAPMethod,
    CLAPOuts,
    _load_clap,
    clap_sim,
)
from aac_metrics.utils.globals import _get_device


class CLAPSim(AACMetric[Union[CLAPOuts, Tensor]]):
    """Cosine-similarity of the Contrastive Language-Audio Pretraining (CLAP) embeddings.

    The implementation is based on the msclap pypi package.
    Note: Instances of this class are not pickable.

    - Paper: https://arxiv.org/pdf/2411.00321
    - msclap package: https://pypi.org/project/msclap/

    For more information, see :func:`~aac_metrics.functional.clap_sim.clap_sim`.
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
        clap_method: CLAPMethod = "text",
        clap_model: Union[str, CLAP] = DEFAULT_CLAP_SIM_MODEL,
        device: Union[str, torch.device, None] = "cuda_if_available",
        batch_size: Optional[int] = 32,
        reset_state: bool = True,
        seed: Optional[int] = 42,
        verbose: int = 0,
    ) -> None:
        if clap_method not in CLAP_METHODS:
            msg = f"Invalid argument {clap_method=}. (expected one of {CLAP_METHODS})"
            raise ValueError(msg)

        device = _get_device(device)
        clap_model = _load_clap(
            clap_model=clap_model,
            device=device,
            reset_state=reset_state,
        )

        super().__init__()
        self._return_all_scores = return_all_scores
        self._clap_method: CLAPMethod = clap_method
        self._clap_model = clap_model
        self._device = device
        self._batch_size = batch_size
        self._reset_state = reset_state
        self._seed = seed
        self._verbose = verbose

        self._candidates = []
        self._mult_references = []
        self._audio_paths = []

    def compute(self) -> Union[CLAPOuts, Tensor]:
        return clap_sim(
            candidates=self._candidates,
            mult_references=self._mult_references,
            audio_paths=self._audio_paths,
            clap_method=self._clap_method,
            return_all_scores=self._return_all_scores,
            clap_model=self._clap_model,
            device=self._device,
            batch_size=self._batch_size,
            reset_state=self._reset_state,
            seed=self._seed,
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
        return ("clap_sim",)

    def reset(self) -> None:
        self._candidates = []
        self._mult_references = []
        self._audio_paths = []
        return super().reset()

    def update(
        self,
        candidates: list[str],
        mult_references_or_audio_paths: Union[list[list[str]], list[str]],
    ) -> None:
        self._candidates += candidates
        if self._clap_method == "audio":
            self._mult_references += mult_references_or_audio_paths
        elif self._clap_method == "text":
            self._audio_paths += mult_references_or_audio_paths
        else:
            msg = (
                f"Invalid value {self._clap_method=}. (expected one of {CLAP_METHODS})"
            )
            raise ValueError(msg)

    def __getstate__(self) -> Any:
        raise RuntimeError(f"{self.__class__.__name__} is not pickable.")
