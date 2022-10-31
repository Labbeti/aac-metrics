#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help

# =================================================================
# This code was pulled from https://github.com/tylin/coco-caption
# and refactored for Python 3.
# =================================================================

import logging
import os.path as osp
import subprocess

from subprocess import Popen
from typing import Union

import torch

from torch import Tensor

from aac_metrics.utils.misc import _check_java_path


logger = logging.getLogger(__name__)


METEOR_JAR_FNAME = osp.join("meteor", "meteor-1.5.jar")


def coco_meteor(
    candidates: list[str],
    mult_references: list[list[str]],
    return_all_scores: bool = True,
    cache_path: str = "$HOME/aac-metrics-cache",
    java_path: str = "java",
    java_max_memory: str = "2G",
    verbose: int = 0,
) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
    """Metric for Evaluation of Translation with Explicit ORdering function.

    Paper: https://dl.acm.org/doi/pdf/10.5555/1626355.1626389

    :param candidates: The list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target.
    :param return_all_scores: If True, returns a tuple containing the globals and locals scores.
        Otherwise returns a scalar tensor containing the main global score.
        defaults to True.
    :param cache_path: The path to the external code directory. defaults to "$HOME/aac-metrics-cache".
    :param java_path: The path to the java executable. defaults to "java".
    :param java_max_memory: The maximal java memory used. defaults to "2G".
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple of globals and locals scores or a scalar tensor with the main global score.
    """
    java_path = osp.expandvars(java_path)
    cache_path = osp.expandvars(cache_path)

    # Prepare java execution
    meteor_jar_fpath = osp.join(cache_path, METEOR_JAR_FNAME)
    meteor_command = [
        java_path,
        "-jar",
        f"-Xmx{java_max_memory}",
        meteor_jar_fpath,
        "-",
        "-",
        "-stdio",
        "-l",
        "en",
        "-norm",
    ]

    if not _check_java_path(java_path):
        raise ValueError(
            f"Cannot find java executable with {java_path=} for compute METEOR metric score."
        )
    if not osp.isfile(meteor_jar_fpath):
        raise FileNotFoundError(
            f"Cannot find JAR file '{METEOR_JAR_FNAME}' in directory '{cache_path}' for meteor metric. Maybe run 'aac-metrics-download' before or specify a 'cache_path' directory."
        )

    if len(candidates) != len(mult_references):
        raise ValueError(
            f"Invalid number of candidates and references. (found {len(candidates)=} != {len(mult_references)=})"
        )

    if verbose >= 2:
        logger.debug(
            f"Start METEOR process with command '{' '.join(meteor_command)}'..."
        )

    meteor_process = Popen(
        meteor_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    eval_line = "EVAL"
    for cand, refs in zip(candidates, mult_references):
        stat = __write_line(cand, refs, meteor_process)
        eval_line += " ||| {}".format(stat)

    assert meteor_process.stdin is not None
    if verbose >= 3:
        logger.debug(f"Write line {eval_line=}.")
    meteor_process.stdin.write("{}\n".format(eval_line).encode())
    meteor_process.stdin.flush()
    assert meteor_process.stdout is not None

    # Read scores
    meteor_scores = []
    for _ in range(len(candidates)):
        meteor_scores_i = float(meteor_process.stdout.readline().strip())
        meteor_scores.append(meteor_scores_i)
    meteor_score = float(meteor_process.stdout.readline().strip())

    meteor_process.stdin.close()
    meteor_process.kill()
    meteor_process.wait()

    dtype = torch.float64
    meteor_score = torch.as_tensor(meteor_score, dtype=dtype)
    meteor_scores = torch.as_tensor(meteor_scores, dtype=dtype)

    if return_all_scores:
        global_scores = {
            "meteor": meteor_score,
        }
        local_scores = {
            "meteor": meteor_scores,
        }
        return global_scores, local_scores
    else:
        return meteor_score


def __write_line(candidate: str, references: list[str], meteor_process: Popen) -> str:
    # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
    candidate = candidate.replace("|||", "").replace("  ", " ")
    score_line = " ||| ".join(("SCORE", " ||| ".join(references), candidate))
    assert meteor_process.stdin is not None
    meteor_process.stdin.write("{}\n".format(score_line).encode())
    meteor_process.stdin.flush()
    assert meteor_process.stdout is not None
    return meteor_process.stdout.readline().decode().strip()
