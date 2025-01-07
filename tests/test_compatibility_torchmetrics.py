#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from unittest import TestCase

from torch import Tensor
from torchmetrics.text.bleu import BLEUScore

from aac_metrics.classes.bleu import BLEU


class TestTorchmetricsCompatibility(TestCase):
    # Tests methods
    def test_bleu(self) -> None:
        cands = ["a man is speaking", "birds chirping"]
        mrefs = [
            [
                "man speaks",
                "man is speaking",
                "a man speaks",
                "man talks",
                "someone is talking",
            ],
            ["a bird is chirping"] * 5,
        ]

        n = 2

        bleu_v1 = BLEU(n=n, return_all_scores=False)
        score_v1: Tensor = bleu_v1(cands, mrefs)  # type: ignore

        bleu_v2 = BLEUScore(n_gram=n, smooth=False)
        score_v2 = bleu_v2(cands, mrefs)

        self.assertAlmostEqual(score_v1.item(), score_v2.item())


if __name__ == "__main__":
    unittest.main()
