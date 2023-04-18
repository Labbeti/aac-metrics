#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib
import os.path as osp
import subprocess
import sys
import unittest

from pathlib import Path
from typing import Callable, Dict, List, Tuple
from unittest import TestCase

from torch import Tensor

from aac_metrics.functional.evaluate import evaluate
from aac_metrics.evaluate import load_csv_file


class TestCompareCaptionEvaluationTools(TestCase):
    # Note: "cet" is here an acronym for "caption evaluation tools"

    # Set Up methods
    @classmethod
    def setUpClass(cls) -> None:
        cls.evaluate_metrics_from_lists = cls._import_cet_eval_func()

    @classmethod
    def _import_cet_eval_func(
        cls,
    ) -> Callable[
        [List[str], List[List[str]]],
        Tuple[Dict[str, float], Dict[int, Dict[str, float]]],
    ]:
        cet_path = osp.join(osp.dirname(__file__), "caption-evaluation-tools")

        stanford_fpath = osp.join(
            cet_path,
            "coco_caption",
            "pycocoevalcap",
            "spice",
            "lib",
            "stanford-corenlp-3.6.0.jar",
        )
        if not osp.isfile(stanford_fpath):
            command = "bash get_stanford_models.sh"
            subprocess.check_call(
                command.split(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=osp.join(cet_path, "coco_caption"),
            )

        # Append cet_path to allow imports of "caption" in eval_metrics.py.
        sys.path.append(cet_path)
        # Override cache and tmp dir to avoid outputs in source code.
        spice_module = importlib.import_module("coco_caption.pycocoevalcap.spice.spice")
        spice_module.CACHE_DIR = "/tmp"  # type: ignore
        spice_module.TEMP_DIR = "/tmp"  # type: ignore
        eval_metrics_module = importlib.import_module("eval_metrics")
        evaluate_metrics_from_lists = eval_metrics_module.evaluate_metrics_from_lists
        return evaluate_metrics_from_lists

    # Tests methods
    def test_example_0(self) -> None:
        cands, mrefs = self._get_example_0()
        self._test_with_example(cands, mrefs)

    def test_example_1(self) -> None:
        fpath = Path(__file__).parent.parent.joinpath("data", "example_1.csv")
        cands, mrefs = load_csv_file(fpath)
        self._test_with_example(cands, mrefs)

    def test_example_2(self) -> None:
        fpath = Path(__file__).parent.parent.joinpath("data", "example_2.csv")
        cands, mrefs = load_csv_file(fpath)
        self._test_with_example(cands, mrefs)

    # Others methods
    def _get_example_0(self) -> tuple[list[str], list[list[str]]]:
        cands = [
            "a man is speaking",
            "birds chirping",
            "rain is falling in the background",
        ]
        mrefs = [
            [
                "man speaks",
                "man is speaking",
                "a man speaks",
                "man talks",
                "someone is talking",
            ],
            ["a bird is chirping"] * 5,
            ["heavy rain noise"] * 5,
        ]
        return cands, mrefs

    def _test_with_example(self, cands: list[str], mrefs: list[list[str]]) -> None:
        corpus_scores, _ = evaluate(cands, mrefs, metrics="dcase2020")
        (
            cet_global_scores,
            _cet_sents_scores,
        ) = self.__class__.evaluate_metrics_from_lists(cands, mrefs)

        cet_global_scores = {k.lower(): v for k, v in cet_global_scores.items()}
        cet_global_scores = {
            (k if k != "cider" else "cider_d"): v for k, v in cet_global_scores.items()
        }

        self.assertIsInstance(corpus_scores, dict)
        self.assertIsInstance(cet_global_scores, dict)

        for name, score in corpus_scores.items():
            self.assertIsInstance(score, Tensor, f"Invalid score type for {name=}")
            self.assertEqual(score.ndim, 0, f"Invalid score ndim for {name=}")

        self.assertListEqual(list(corpus_scores.keys()), list(cet_global_scores.keys()))
        for metric_name, v1 in corpus_scores.items():
            v1 = v1.item()
            v2 = cet_global_scores[metric_name]
            self.assertEqual(v1, v2, f"{metric_name=}")


if __name__ == "__main__":
    unittest.main()
