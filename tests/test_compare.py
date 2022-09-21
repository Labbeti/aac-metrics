#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib
import os.path as osp
import subprocess
import sys
import unittest

from typing import Callable, Dict, List, Tuple
from unittest import TestCase

from aac_metrics.evaluate import evaluate


class TestCompare(TestCase):
    def init_caption_evaluation_tools(self) -> None:
        cet_path = osp.join(osp.dirname(__file__), "caption-evaluation-tools")

        stanford_fpath = osp.join(
            cet_path, "coco_caption", "tokenizer", "stanford-corenlp-3.4.1.jar"
        )
        if not osp.isfile(stanford_fpath):
            command = "bash get_stanford_models.sh"
            subprocess.check_call(
                command.split(),
                cwd=osp.join(cet_path, "coco_caption"),
            )

    def get_eval_function(
        self,
    ) -> Callable[
        [List[str], List[List[str]]],
        Tuple[Dict[str, float], Dict[int, Dict[str, float]]],
    ]:
        cet_path = osp.join(osp.dirname(__file__), "caption-evaluation-tools")
        self.init_caption_evaluation_tools()
        # Append cet_path to allow imports of "coco_caption" in eval_metrics.py.
        sys.path.append(cet_path)
        # Override cache and tmp dir to avoid outputs in source code.
        spice_module = importlib.import_module("coco_caption.pycocoevalcap.spice.spice")
        spice_module.CACHE_DIR = "/tmp"  # type: ignore
        spice_module.TEMP_DIR = "/tmp"  # type: ignore
        eval_metrics_module = importlib.import_module("eval_metrics")
        evaluate_metrics_from_lists = eval_metrics_module.evaluate_metrics_from_lists
        return evaluate_metrics_from_lists

    def test_compare_results(self) -> None:
        evaluate_metrics_from_lists = self.get_eval_function()
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

        global_scores, _ = evaluate(cands, mrefs)
        cet_global_scores, _cet_local_scores = evaluate_metrics_from_lists(cands, mrefs)

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


if __name__ == "__main__":
    unittest.main()
