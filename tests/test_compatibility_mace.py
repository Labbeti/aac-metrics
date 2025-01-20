#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib
import logging
import random
import sys
import unittest
from functools import partial
from pathlib import Path
from typing import Callable
from unittest import TestCase

import torch

from aac_metrics.classes.mace import MACE

pylog = logging.getLogger(__name__)


class TestMACECompatibility(TestCase):
    # Set Up methods
    @classmethod
    def setUpClass(cls) -> None:
        method = "combined"
        echecker = "echecker_clotho_audiocaps_base"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # device = "cpu"

        src_mace_fn = cls._get_src_mace()
        src_mace_fn = partial(src_mace_fn, method=method, echecker=echecker)

        cls.new_mace = MACE(
            mace_method=method,
            return_all_scores=True,
            device=device,
            verbose=2,
            echecker=echecker,
            seed=None,  # seed will be set just before testing to ensure same seed for src and new MACE
        )
        cls.src_mace = src_mace_fn

    @classmethod
    def _get_src_mace(cls) -> Callable:
        mace_dpath = Path(__file__).parent.joinpath("mace", "mace_metric")
        sys.path.append(str(mace_dpath))
        mace_module = importlib.import_module("mace")
        mace_fn = mace_module.mace
        return mace_fn

    # Tests methods
    def test_examples_mace(self) -> None:
        audio_dpath = Path(__file__).parent.joinpath("mace", "assets")
        candidates = ["a man is speaking", "rain falls"]
        mult_references = [
            [
                "a man speaks.",
                "someone speaks.",
                "a man is speaking while a bird is chirping in the background",
            ],
            ["rain is falling hard on a surface"],
        ]
        audio_paths = [
            audio_dpath.joinpath("woman_singing.wav"),
            audio_dpath.joinpath("rain.wav"),
        ]
        audio_paths = list(map(str, audio_paths))

        seed = random.randint(0, 2**16)

        random.seed(seed)
        new_outs = self.new_mace(
            candidates=candidates,
            mult_references=mult_references,
            audio_paths=audio_paths,
        )

        random.seed(seed)
        src_outs = self.src_mace(
            candidates=candidates,
            mult_references=mult_references,
            audio_paths=audio_paths,
        )

        for new_dict, src_dict in zip(new_outs, src_outs):
            assert isinstance(new_dict, dict)
            assert isinstance(src_dict, dict)
            assert set(new_dict.keys()) == set(src_dict.keys())

            equals = {k: torch.equal(new_dict[k], src_dict[k]) for k in new_dict.keys()}
            assert all(equals.values()), f"{equals=}; {new_dict=}; {src_dict=}"


if __name__ == "__main__":
    unittest.main()
