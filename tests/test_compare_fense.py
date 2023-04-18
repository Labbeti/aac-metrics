#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib
import os.path as osp
import sys
import torch
import unittest

from typing import Any
from unittest import TestCase

from aac_metrics.classes.sbert_sim import SBERTSim
from aac_metrics.classes.fense import FENSE
from aac_metrics.evaluate import load_csv_file


class TestCompareFENSE(TestCase):
    # Set Up methods
    @classmethod
    def setUpClass(cls) -> None:
        Evaluator = cls._get_src_evaluator_class()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device=}")

        cls.src_sbert_sim = Evaluator(
            device=device,
            echecker_model="none",
        )
        cls.src_fense = Evaluator(
            device=device,
            echecker_model="echecker_clotho_audiocaps_base",
        )

        cls.new_sbert_sim = SBERTSim(
            return_all_scores=False,
            device=device,
            verbose=2,
        )
        cls.new_fense = FENSE(
            return_all_scores=False,
            device=device,
            verbose=2,
            echecker="echecker_clotho_audiocaps_base",
        )

    @classmethod
    def _get_src_evaluator_class(cls) -> Any:
        fense_path = osp.join(osp.dirname(__file__), "fense")
        sys.path.append(fense_path)
        fense_module = importlib.import_module("fense.evaluator")
        return fense_module.Evaluator

    # Tests methods
    def test_example_1_fense(self) -> None:
        fpath = osp.join(osp.dirname(__file__), "..", "data", "example_1.csv")
        self._test_with_original_fense(fpath)

    def test_example_1_sbert_sim(self) -> None:
        fpath = osp.join(osp.dirname(__file__), "..", "data", "example_1.csv")
        self._test_with_original_sbert_sim(fpath)

    def test_example_2_fense(self) -> None:
        fpath = osp.join(osp.dirname(__file__), "..", "data", "example_2.csv")
        self._test_with_original_fense(fpath)

    def test_example_2_sbert_sim(self) -> None:
        fpath = osp.join(osp.dirname(__file__), "..", "data", "example_2.csv")
        self._test_with_original_sbert_sim(fpath)

    def test_output_size(self) -> None:
        fpath = osp.join(osp.dirname(__file__), "..", "data", "example_1.csv")
        cands, mrefs = load_csv_file(fpath)

        self.new_fense._return_all_scores = True
        corpus_scores, sents_scores = self.new_fense(cands, mrefs)
        self.new_fense._return_all_scores = False

        for name, score in corpus_scores.items():
            self.assertEqual(score.ndim, 0)

        for name, scores in sents_scores.items():
            self.assertEqual(scores.ndim, 1)

        for name, scores in sents_scores.items():
            self.assertEqual(len(cands), len(scores), f"{name=}")

    # Others methods
    def _test_with_original_fense(self, fpath: str) -> None:
        cands, mrefs = load_csv_file(fpath)

        src_fense_score = self.src_fense.corpus_score(cands, mrefs).item()
        new_fense_score = self.new_fense(cands, mrefs).item()

        print(f"{fpath=}")
        print(f"{src_fense_score=}")
        print(f"{new_fense_score=}")

        self.assertEqual(
            src_fense_score,
            new_fense_score,
            "Invalid FENSE score with original implementation.",
        )

    def _test_with_original_sbert_sim(self, fpath: str) -> None:
        cands, mrefs = load_csv_file(fpath)

        src_sbert_sim_score = self.src_sbert_sim.corpus_score(cands, mrefs).item()
        new_sbert_sim_score = self.new_sbert_sim(cands, mrefs).item()

        print(f"{fpath=}")
        print(f"{src_sbert_sim_score=}")
        print(f"{new_sbert_sim_score=}")

        self.assertEqual(
            src_sbert_sim_score,
            new_sbert_sim_score,
            "Invalid SBERTSim score with original implementation.",
        )


if __name__ == "__main__":
    unittest.main()
