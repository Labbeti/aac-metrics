#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from unittest import TestCase

from aac_metrics.classes.bleu import BLEU
from aac_metrics.utils.imports import _TORCHMETRICS_AVAILABLE

if _TORCHMETRICS_AVAILABLE:
    from torchmetrics.text.bleu import BLEUScore


class TestBleu(TestCase):
    def test_bleu(self) -> None:
        if not _TORCHMETRICS_AVAILABLE:
            return None

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
        score_v1 = bleu_v1(cands, mrefs)

        bleu_v2 = BLEUScore(n_gram=n, smooth=False)
        score_v2 = bleu_v2(cands, mrefs)

        self.assertAlmostEqual(score_v1.item(), score_v2.item())


if __name__ == "__main__":
    unittest.main()
