#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib
import logging
import os.path as osp
import sys
import torch
import unittest

from typing import Any
from unittest import TestCase

import transformers

from aac_metrics.classes.fense import FENSE
from aac_metrics.functional.fer import _use_new_echecker_loading
from aac_metrics.eval import load_csv_file


pylog = logging.getLogger(__name__)


class TestCompareFENSE(TestCase):
    # Set Up methods
    @classmethod
    def setUpClass(cls) -> None:
        Evaluator = cls._get_src_evaluator_class()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device=}")

        echecker = "echecker_clotho_audiocaps_base"
        cls.new_fense = FENSE(
            return_all_scores=True,
            device=device,
            verbose=2,
            echecker=echecker,
        )
        cls.src_sbert_sim = Evaluator(
            device=device,
            echecker_model="none",
        )
        if _use_new_echecker_loading():
            cls.src_fense = None
        else:
            cls.src_fense = Evaluator(
                device=device,
                echecker_model=echecker,
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

    def test_example_2_fense(self) -> None:
        fpath = osp.join(osp.dirname(__file__), "..", "data", "example_2.csv")
        self._test_with_original_fense(fpath)

    def test_output_size(self) -> None:
        fpath = osp.join(osp.dirname(__file__), "..", "data", "example_1.csv")
        cands, mrefs = load_csv_file(fpath)

        self.new_fense._return_all_scores = True
        outs: tuple = self.new_fense(cands, mrefs)  # type: ignore
        corpus_scores, sents_scores = outs
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

        src_sbert_sim_score = self.src_sbert_sim.corpus_score(cands, mrefs).item()

        outs: tuple = self.new_fense(cands, mrefs)  # type: ignore
        corpus_outs, _sents_outs = outs
        new_sbert_sim_score = corpus_outs["sbert_sim"].item()
        new_fense_score = corpus_outs["fense"].item()

        self.assertEqual(
            src_sbert_sim_score,
            new_sbert_sim_score,
            "Invalid SBERTSim score with original implementation.",
        )

        if self.src_fense is None:
            pylog.warning(
                f"Skipping test with original FENSE for the transformers version {transformers.__version__}"
            )
        else:
            src_fense_score = self.src_fense.corpus_score(cands, mrefs).item()
            self.assertEqual(
                src_fense_score,
                new_fense_score,
                "Invalid FENSE score with original implementation.",
            )


if __name__ == "__main__":
    unittest.main()
