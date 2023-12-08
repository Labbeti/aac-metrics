#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from unittest import TestCase

import torch

from torch import Tensor

from aac_metrics.classes.spider import SPIDEr
from aac_metrics.classes.spider_max import SPIDErMax


class TestSPIDErMax(TestCase):
    # Tests methods
    def test_sd_vs_sdmax(self) -> None:
        sd = SPIDEr(return_all_scores=False)
        sdmax = SPIDErMax(return_all_scores=False)

        cands, mrefs = self._get_example_0()
        mcands = [[cand] for cand in cands]

        sd_score = sd(cands, mrefs)
        sdmax_score = sdmax(mcands, mrefs)

        assert isinstance(sd_score, Tensor)
        assert isinstance(sdmax_score, Tensor)
        self.assertTrue(
            torch.allclose(sd_score, sdmax_score), f"{sd_score=}, {sdmax_score=}"
        )

    def _get_example_0(self) -> tuple[list[str], list[list[str]]]:
        cands = [
            "a man is speaking",
            "birds chirping",
            "rain is falling in the background",
        ]
        mrefs = [
            [
                "man speaks",
                "man is speaking",
                "a man speaks",
                "man talks",
                "someone is talking",
            ],
            ["a bird is chirping"] * 5,
            ["heavy rain noise"] * 5,
        ]
        return cands, mrefs


if __name__ == "__main__":
    unittest.main()
