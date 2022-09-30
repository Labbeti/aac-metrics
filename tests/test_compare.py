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

from aac_metrics.evaluate import aac_evaluate, load_csv_file


class TestCompare(TestCase):
    # Note: "cet" is here an acronym for "caption evaluation tools"

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.evaluate_metrics_from_lists = self.import_cet_eval_func()

    def install_spice(self) -> None:
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
                cwd=osp.join(cet_path, "coco_caption"),
            )

    def import_cet_eval_func(
        self,
    ) -> Callable[
        [List[str], List[List[str]]],
        Tuple[Dict[str, float], Dict[int, Dict[str, float]]],
    ]:
        cet_path = osp.join(osp.dirname(__file__), "caption-evaluation-tools")
        self.install_spice()
        # Append cet_path to allow imports of "coco_caption" in eval_metrics.py.
        sys.path.append(cet_path)
        # Override cache and tmp dir to avoid outputs in source code.
        spice_module = importlib.import_module("coco_caption.pycocoevalcap.spice.spice")
        spice_module.CACHE_DIR = "/tmp"  # type: ignore
        spice_module.TEMP_DIR = "/tmp"  # type: ignore
        eval_metrics_module = importlib.import_module("eval_metrics")
        evaluate_metrics_from_lists = eval_metrics_module.evaluate_metrics_from_lists
        return evaluate_metrics_from_lists

    def get_example_0(self) -> tuple[list[str], list[list[str]]]:
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

    def test_example_0(self) -> None:
        cands, mrefs = self.get_example_0()
        global_scores, _ = aac_evaluate(cands, mrefs)
        cet_global_scores, _cet_local_scores = self.evaluate_metrics_from_lists(
            cands, mrefs
        )

        cet_global_scores = {k.lower(): v for k, v in cet_global_scores.items()}
        cet_global_scores = {
            (k if k != "cider" else "cider_d"): v for k, v in cet_global_scores.items()
        }

        self.assertIsInstance(global_scores, dict)
        self.assertIsInstance(cet_global_scores, dict)
        self.assertListEqual(list(global_scores.keys()), list(cet_global_scores.keys()))
        for metric_name, v1 in global_scores.items():
            v2 = cet_global_scores[metric_name]
            self.assertEqual(v1.item(), v2, f"{metric_name=}")
        self.assertDictEqual(global_scores, cet_global_scores)

    def test_example_1(self) -> None:
        fpath = Path(__file__).parent.parent.joinpath("examples", "example_1.csv")
        candidates, mult_references = load_csv_file(fpath)

        global_scores, _ = aac_evaluate(candidates, mult_references)
        cet_global_scores, _cet_local_scores = self.evaluate_metrics_from_lists(
            candidates, mult_references
        )

        cet_global_scores = {k.lower(): v for k, v in cet_global_scores.items()}
        cet_global_scores = {
            (k if k != "cider" else "cider_d"): v for k, v in cet_global_scores.items()
        }

        self.assertIsInstance(global_scores, dict)
        self.assertIsInstance(cet_global_scores, dict)
        self.assertListEqual(list(global_scores.keys()), list(cet_global_scores.keys()))
        for metric_name, v1 in global_scores.items():
            v2 = cet_global_scores[metric_name]
            self.assertEqual(v1.item(), v2, f"{metric_name=}")

    def test_example_2(self) -> None:
        fpath = Path(__file__).parent.parent.joinpath("examples", "example_2.csv")
        candidates, mult_references = load_csv_file(fpath)

        global_scores, _ = aac_evaluate(candidates, mult_references)
        cet_global_scores, _cet_local_scores = self.evaluate_metrics_from_lists(
            candidates, mult_references
        )

        cet_global_scores = {k.lower(): v for k, v in cet_global_scores.items()}
        cet_global_scores = {
            (k if k != "cider" else "cider_d"): v for k, v in cet_global_scores.items()
        }

        self.assertIsInstance(global_scores, dict)
        self.assertIsInstance(cet_global_scores, dict)
        self.assertListEqual(list(global_scores.keys()), list(cet_global_scores.keys()))
        for metric_name, v1 in global_scores.items():
            v2 = cet_global_scores[metric_name]
            self.assertEqual(v1.item(), v2, f"{metric_name=}")


if __name__ == "__main__":
    unittest.main()
