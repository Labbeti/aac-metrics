#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from unittest import TestCase

try:
    from torchmetrics.text.bleu import BLEUScore

    TORCHMETRICS_IS_INSTALLED = True
except ModuleNotFoundError:
    TORCHMETRICS_IS_INSTALLED = False

from aac_metrics.classes.coco_bleu import CocoBLEU


class TestBleu(TestCase):
    def test_bleu(self) -> None:
        if not TORCHMETRICS_IS_INSTALLED:
            return
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
        tokenizer = str.split
        tok_cands = list(map(tokenizer, cands))
        tok_mrefs = [list(map(tokenizer, refs)) for refs in mrefs]

        bleu_v1 = CocoBLEU(n=n, return_all_scores=False)
        score_v1 = bleu_v1(cands, mrefs)

        bleu_v2 = BLEUScore(n_gram=n, smooth=False)
        score_v2 = bleu_v2(tok_mrefs, tok_cands)

        self.assertAlmostEqual(score_v1.item(), score_v2.item())


if __name__ == "__main__":
    unittest.main()
