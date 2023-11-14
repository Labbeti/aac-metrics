#!/usr/bin/env python
# -*- coding: utf-8 -*-

import platform
import unittest

from unittest import TestCase

from aac_metrics.functional.evaluate import evaluate


class TestAll(TestCase):
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

        _ = evaluate(cands, mrefs, metrics="all")


if __name__ == "__main__":
    unittest.main()
