#!/usr/bin/env python
# -*- coding: utf-8 -*-

import importlib
import os
import os.path as osp
import shutil
import subprocess
import unittest

from typing import Callable, Dict, List, Tuple
from unittest import TestCase

from aac_metrics.evaluate import evaluate


GIT_LINK = "https://github.com/audio-captioning/caption-evaluation-tools/"


class TestCompare(TestCase):
    def init_caption_evaluation_tools(self) -> None:
        git_dpath = osp.join(osp.dirname(__file__), "caption-evaluation-tools")
        if osp.exists(git_dpath):
            shutil.rmtree(git_dpath)

        exitcode = subprocess.check_call(["git", "--version"])
        if exitcode != 0:
            raise ValueError("Cannot use git in test.")

        command = f"git clone {GIT_LINK} {git_dpath}"
        exitcode = subprocess.check_call(command.split())

        for name in os.listdir(git_dpath):
            tgt_parent = osp.dirname(git_dpath)
            tgt_path = osp.join(tgt_parent, name)
            if osp.isdir(tgt_path):
                shutil.rmtree(tgt_path)
            elif osp.isfile(tgt_path):
                os.remove(tgt_path)
            shutil.move(osp.join(git_dpath, name), tgt_parent)

        os.rmdir(git_dpath)

        command = "bash get_stanford_models.sh"
        exitcode = subprocess.check_call(
            command.split(), cwd=osp.join(osp.dirname(__file__), "coco_caption")
        )

    def get_eval_function(
        self,
    ) -> Callable[
        [List[str], List[List[str]]],
        Tuple[Dict[str, float], Dict[int, Dict[str, float]]],
    ]:
        try:
            module = importlib.import_module("eval_metrics")
            evaluate_metrics_from_lists = module.evaluate_metrics_from_lists
        except (ModuleNotFoundError, AttributeError):
            self.init_caption_evaluation_tools()
            module = importlib.import_module("eval_metrics")
            evaluate_metrics_from_lists = module.evaluate_metrics_from_lists
        return evaluate_metrics_from_lists

    def test_compare_results(self) -> None:
        evaluate_metrics_from_lists = self.get_eval_function()
        print(evaluate_metrics_from_lists)

        cands = ["a man is speaking", "birds chirping"]
        mrefs = [
            [
                "man speaks",
                "man is speaking",
                "a man speaks",
                "man talks",
                "someone is talking",
            ],
            ["a bird is chirping"] * 5,
        ]

        global_scores, _ = evaluate(cands, mrefs)
        cet_global_scores, _cet_local_scores = evaluate_metrics_from_lists(cands, mrefs)

        cet_global_scores = {k.lower(): v for k, v in cet_global_scores.items()}
        cet_global_scores = {(k if k != "cider" else "cider_d"): v for k, v in cet_global_scores.items()}

        self.assertIsInstance(global_scores, dict)
        self.assertIsInstance(cet_global_scores, dict)
        self.assertListEqual(list(global_scores.keys()), list(cet_global_scores.keys()))
        for k, v1 in global_scores.items():
            v2 = cet_global_scores[k]
            self.assertEqual(v1, v2)
        self.assertDictEqual(global_scores, cet_global_scores)


if __name__ == "__main__":
    unittest.main()
