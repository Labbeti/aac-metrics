#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from unittest import TestCase

import torch

from aac_metrics import evaluate
from aac_metrics.functional import cider_d
from aac_metrics.utils.tokenization import (
    preprocess_mono_sents,
    preprocess_mult_sents,
)


class TestReadmeExamples(TestCase):
    def test_example_1(self) -> None:
        candidates: list[str] = ["a man is speaking", "rain falls"]
        mult_references: list[list[str]] = [
            [
                "a man speaks.",
                "someone speaks.",
                "a man is speaking while a bird is chirping in the background",
            ],
            ["rain is falling hard on a surface"],
        ]

        corpus_scores, _ = evaluate(candidates, mult_references)
        # print(corpus_scores)
        # dict containing the score of each metric: "bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_l", "meteor", "cider_d", "spice", "spider"
        # {"bleu_1": tensor(0.4278), "bleu_2": ..., ...}

        expected_keys = [
            "bleu_1",
            "bleu_2",
            "bleu_3",
            "bleu_4",
            "rouge_l",
            "meteor",
            "cider_d",
            "spice",
            "spider",
        ]
        self.assertSetEqual(set(corpus_scores.keys()), set(expected_keys))
        self.assertTrue(
            torch.allclose(
                corpus_scores["bleu_1"],
                torch.as_tensor(0.4278, dtype=torch.float64),
                atol=0.0001,
            ),
            f"{corpus_scores['bleu_1']=}",
        )

    def test_example_2(self) -> None:
        candidates: list[str] = ["a man is speaking", "rain falls"]
        mult_references: list[list[str]] = [
            [
                "a man speaks.",
                "someone speaks.",
                "a man is speaking while a bird is chirping in the background",
            ],
            ["rain is falling hard on a surface"],
        ]

        corpus_scores, _ = evaluate(candidates, mult_references, metrics="dcase2023")
        # print(corpus_scores)
        # dict containing the score of each metric: "meteor", "cider_d", "spice", "spider", "spider_fl", "fer"

        expected_keys = ["meteor", "cider_d", "spice", "spider", "spider_fl", "fer"]
        self.assertTrue(set(corpus_scores.keys()).issuperset(expected_keys))

    def test_example_3(self) -> None:
        candidates: list[str] = ["a man is speaking", "rain falls"]
        mult_references: list[list[str]] = [
            [
                "a man speaks.",
                "someone speaks.",
                "a man is speaking while a bird is chirping in the background",
            ],
            ["rain is falling hard on a surface"],
        ]

        candidates = preprocess_mono_sents(candidates)
        mult_references = preprocess_mult_sents(mult_references)

        outputs: tuple[dict, dict] = cider_d(candidates, mult_references)  # type: ignore
        corpus_scores, sents_scores = outputs
        # print(corpus_scores)
        # {"cider_d": tensor(0.9614)}
        # print(sents_scores)
        # {"cider_d": tensor([1.3641, 0.5587])}

        self.assertTrue(set(corpus_scores.keys()).issuperset({"cider_d"}))
        self.assertTrue(set(sents_scores.keys()).issuperset({"cider_d"}))

        dtype = torch.float64

        self.assertTrue(
            torch.allclose(
                corpus_scores["cider_d"],
                torch.as_tensor(0.9614, dtype=dtype),
                atol=0.0001,
            )
        )
        self.assertTrue(
            torch.allclose(
                sents_scores["cider_d"],
                torch.as_tensor([1.3641, 0.5587], dtype=dtype),
                atol=0.0001,
            )
        )


if __name__ == "__main__":
    unittest.main()
