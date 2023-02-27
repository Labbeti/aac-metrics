#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ORIGINAL CODE FROM https://github.com/tylin/coco-caption

import logging
import os.path as osp
import subprocess

from subprocess import Popen
from typing import Union

import torch

from torch import Tensor

from aac_metrics.utils.checks import check_java_path


logger = logging.getLogger(__name__)


FNAME_METEOR_JAR = osp.join("aac-metrics", "meteor", "meteor-1.5.jar")
SUPPORTED_LANGUAGES = ("en", "cz", "de", "es", "fr")


def meteor(
    candidates: list[str],
    mult_references: list[list[str]],
    return_all_scores: bool = True,
    cache_path: str = "$HOME/.cache",
    java_path: str = "java",
    java_max_memory: str = "2G",
    language: str = "en",
    verbose: int = 0,
) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
    """Metric for Evaluation of Translation with Explicit ORdering function.

    - Paper: https://dl.acm.org/doi/pdf/10.5555/1626355.1626389
    - Documentation: https://www.cs.cmu.edu/~alavie/METEOR/README.html

    :param candidates: The list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target.
    :param return_all_scores: If True, returns a tuple containing the globals and locals scores.
        Otherwise returns a scalar tensor containing the main global score.
        defaults to True.
    :param cache_path: The path to the external code directory. defaults to "$HOME/.cache".
    :param java_path: The path to the java executable. defaults to "java".
    :param java_max_memory: The maximal java memory used. defaults to "2G".
    :param language: The language used for stem, synonym and paraphrase matching. defaults to "en".
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple of globals and locals scores or a scalar tensor with the main global score.
    """
    cache_path = osp.expandvars(cache_path)
    java_path = osp.expandvars(java_path)

    meteor_jar_fpath = osp.join(cache_path, FNAME_METEOR_JAR)
    language = "en"  # supported: en cz de es fr

    if __debug__:
        if not osp.isfile(meteor_jar_fpath):
            raise FileNotFoundError(
                f"Cannot find JAR file '{meteor_jar_fpath}' for METEOR metric. Maybe run 'aac-metrics-download' or specify another 'cache_path' directory."
            )
        if not check_java_path(java_path):
            raise ValueError(
                f"Cannot find java executable with {java_path=} for compute METEOR metric score."
            )

    if len(candidates) != len(mult_references):
        raise ValueError(
            f"Invalid number of candidates and references. (found {len(candidates)=} != {len(mult_references)=})"
        )

    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Invalid argument {language=}. (expected one of {SUPPORTED_LANGUAGES})"
        )

    meteor_cmd = [
        java_path,
        "-jar",
        f"-Xmx{java_max_memory}",
        meteor_jar_fpath,
        "-",
        "-",
        "-stdio",
        "-l",
        language,
        "-norm",
    ]

    if verbose >= 2:
        logger.debug(f"Start METEOR process with command '{' '.join(meteor_cmd)}'...")

    meteor_process = Popen(
        meteor_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    n_candidates = len(candidates)
    encoded_cands_and_mrefs = [
        encore_cand_and_refs(cand, refs)
        for cand, refs in zip(candidates, mult_references)
    ]
    del candidates, mult_references

    # Encode candidates and references
    eval_line = "EVAL"
    for encoded in encoded_cands_and_mrefs:
        assert meteor_process.stdin is not None, "INTERNAL METEOR process error"
        meteor_process.stdin.write(encoded)
        meteor_process.stdin.flush()
        assert meteor_process.stdout is not None, "INTERNAL METEOR process error"
        stat = meteor_process.stdout.readline().decode().strip()
        eval_line += " ||| {}".format(stat)

    # Eval encoded candidates and references
    assert meteor_process.stdin is not None, "INTERNAL METEOR process error"
    if verbose >= 3:
        logger.debug(f"Write line {eval_line=}.")
    meteor_process.stdin.write("{}\n".format(eval_line).encode())
    meteor_process.stdin.flush()

    # Read scores
    assert meteor_process.stdout is not None, "INTERNAL METEOR process error"
    meteor_scores = []
    for _ in range(n_candidates):
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
        corpus_scores = {
            "meteor": meteor_score,
        }
        sents_scores = {
            "meteor": meteor_scores,
        }
        return corpus_scores, sents_scores
    else:
        return meteor_score


def encore_cand_and_refs(candidate: str, references: list[str]) -> bytes:
    # SCORE ||| reference 1 words ||| ... ||| reference N words ||| candidate words
    candidate = candidate.replace("|||", "").replace("  ", " ")
    score_line = " ||| ".join(("SCORE", " ||| ".join(references), candidate))
    encoded = "{}\n".format(score_line).encode()
    return encoded
