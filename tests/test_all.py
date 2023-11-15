#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from unittest import TestCase

from aac_metrics.functional.evaluate import evaluate


class TestAll(TestCase):
    def test_example_1(self) -> None:
        cands: list[str] = ["a man is speaking", "rain falls"]
        mrefs: list[list[str]] = [
            [
                "a man speaks.",
                "someone speaks.",
                "a man is speaking while a bird is chirping in the background",
            ],
            ["rain is falling hard on a surface"],
        ]

        _ = evaluate(cands, mrefs, metrics="all")


if __name__ == "__main__":
    unittest.main()
