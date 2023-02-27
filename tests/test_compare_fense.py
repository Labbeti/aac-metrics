#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib
import os.path as osp
import sys
import torch
import unittest

from typing import Any
from unittest import TestCase

from aac_metrics.classes.fense import FENSE
from aac_metrics.evaluate import load_csv_file


class TestCompareFENSE(TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"{device=}")
        Evaluator = self.get_orig_evaluator_class()
        self.evaluator = Evaluator(
            device=device,
            echecker_model="echecker_clotho_audiocaps_base",
        )
        self.my_fense = FENSE(
            return_all_scores=False,
            device=device,
            verbose=2,
            echecker="echecker_clotho_audiocaps_base",
        )

    def get_orig_evaluator_class(self) -> Any:
        fense_path = osp.join(osp.dirname(__file__), "fense")
        sys.path.append(fense_path)
        fense_module = importlib.import_module("fense.evaluator")
        return fense_module.Evaluator

    def test_example_1(self) -> None:
        fpath = osp.join(osp.dirname(__file__), "..", "examples", "example_1.csv")
        self._test_with_original_fense(fpath)

    def test_example_2(self) -> None:
        fpath = osp.join(osp.dirname(__file__), "..", "examples", "example_2.csv")
        self._test_with_original_fense(fpath)

    def test_output_size(self) -> None:
        fpath = osp.join(osp.dirname(__file__), "..", "examples", "example_1.csv")
        cands, mrefs = load_csv_file(fpath)

        self.my_fense._return_all_scores = True
        _global_scores, sents_scores = self.my_fense(cands, mrefs)
        self.my_fense._return_all_scores = False

        for name, sents_score in sents_scores.items():
            print(f"{name=}, {sents_score.shape=}")
            self.assertEqual(len(cands), len(sents_score), f"{name=}")

    def _test_with_original_fense(self, fpath: str) -> None:
        cands, mrefs = load_csv_file(fpath)

        orig_score = self.evaluator.corpus_score(cands, mrefs).item()
        score = self.my_fense(cands, mrefs).item()

        print(f"{orig_score=} ({type(orig_score)=})")
        print(f"{score=} ({type(score)=})")

        self.assertEqual(orig_score, score)


if __name__ == "__main__":
    unittest.main()
