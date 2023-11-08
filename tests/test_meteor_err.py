#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from unittest import TestCase

from aac_metrics.functional.meteor import meteor


class TestCompare(TestCase):
    def test_example_1(self) -> None:
        cands = ["many green trees are in two sides of a curved green river."]
        mrefs = [
            [
                "many green trees are in two sides of a curved green river .",
                "many green trees are in two sides of a curved green river .",
                "many green trees are in two sides of a curved green river .",
                "many green trees are in two sides of a curved green river .",
                "many green trees are in two sides of a curved green river .",
            ]
        ]
        meteor_outs_corpus, _ = meteor(cands, mrefs)
        score = meteor_outs_corpus["meteor"].item()  # type: ignore
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
