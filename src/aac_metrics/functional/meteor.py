#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os.path as osp
import platform
import subprocess

from pathlib import Path
from subprocess import Popen
from typing import Iterable, Optional, Union

import torch

from torch import Tensor

from aac_metrics.utils.checks import check_java_path, check_metric_inputs
from aac_metrics.utils.globals import _get_cache_path, _get_java_path


pylog = logging.getLogger(__name__)


DNAME_METEOR_CACHE = osp.join("aac-metrics", "meteor")
FNAME_METEOR_JAR = osp.join(DNAME_METEOR_CACHE, "meteor-1.5.jar")
SUPPORTED_LANGUAGES = ("en", "cz", "de", "es", "fr")


def meteor(
    candidates: list[str],
    mult_references: list[list[str]],
    return_all_scores: bool = True,
    cache_path: Union[str, Path, None] = None,
    java_path: Union[str, Path, None] = None,
    java_max_memory: str = "2G",
    language: str = "en",
    use_shell: Optional[bool] = None,
    params: Optional[Iterable[float]] = None,
    weights: Optional[Iterable[float]] = None,
    verbose: int = 0,
) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
    """Metric for Evaluation of Translation with Explicit ORdering function.

    - Paper: https://dl.acm.org/doi/pdf/10.5555/1626355.1626389
    - Documentation: https://www.cs.cmu.edu/~alavie/METEOR/README.html
    - Original implementation: https://github.com/tylin/coco-caption

    :param candidates: The list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target.
    :param return_all_scores: If True, returns a tuple containing the globals and locals scores.
        Otherwise returns a scalar tensor containing the main global score.
        defaults to True.
    :param cache_path: The path to the external code directory. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_cache_path`.
    :param java_path: The path to the java executable. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_java_path`.
    :param java_max_memory: The maximal java memory used. defaults to "2G".
    :param language: The language used for stem, synonym and paraphrase matching.
        Can be one of ("en", "cz", "de", "es", "fr").
        defaults to "en".
    :param use_shell: Optional argument to force use os-specific shell for the java subprogram.
        If None, it will use shell only on Windows OS.
        defaults to None.
    :param params: List of 4 parameters (alpha, beta gamma delta) used in METEOR metric.
        If None, it will use the default of the java program, which is (0.85, 0.2, 0.6, 0.75).
        defaults to None.
    :param weights: List of 4 parameters (w1, w2, w3, w4) used in METEOR metric.
        If None, it will use the default of the java program, which is (1.0 1.0 0.6 0.8).
        defaults to None.
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple of globals and locals scores or a scalar tensor with the main global score.
    """
    check_metric_inputs(candidates, mult_references)

    cache_path = _get_cache_path(cache_path)
    java_path = _get_java_path(java_path)

    meteor_jar_fpath = osp.join(cache_path, FNAME_METEOR_JAR)

    if use_shell is None:
        use_shell = platform.system() == "Windows"

    if __debug__:
        if not osp.isfile(meteor_jar_fpath):
            raise FileNotFoundError(
                f"Cannot find JAR file '{meteor_jar_fpath}' for METEOR metric. Maybe run 'aac-metrics-download' or specify another 'cache_path' directory."
            )
        if not check_java_path(java_path):
            raise RuntimeError(
                f"Invalid Java executable to compute METEOR score. ({java_path})"
            )

    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Invalid argument {language=}. (expected one of {SUPPORTED_LANGUAGES})"
        )

    # Note: override localization to avoid errors due to double conversion (https://github.com/Labbeti/aac-metrics/issues/9)
    meteor_cmd = [
        java_path,
        "-Duser.country=US",
        "-Duser.language=en",
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

    if params is not None:
        params = list(params)
        if len(params) != 4:
            raise ValueError(
                f"Invalid argument {params=}. (expected 4 params but found {len(params)})"
            )
        params_arg = " ".join(map(str, params))
        meteor_cmd += ["-p", f"{params_arg}"]

    if weights is not None:
        weights = list(weights)
        if len(weights) != 4:
            raise ValueError(
                f"Invalid argument {weights=}. (expected 4 params but found {len(weights)})"
            )
        weights_arg = " ".join(map(str, weights))
        meteor_cmd += ["-w", f"{weights_arg}"]

    if verbose >= 2:
        pylog.debug(
            f"Run METEOR java code with: {' '.join(meteor_cmd)} and {use_shell=}"
        )

    meteor_process = Popen(
        meteor_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=use_shell,
    )

    n_candidates = len(candidates)
    encoded_cands_and_mrefs = [
        _encode_cand_and_refs(cand, refs)
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
        pylog.debug(f"Write line {eval_line=}.")

    process_inputs = "{}\n".format(eval_line).encode()
    meteor_process.stdin.write(process_inputs)
    meteor_process.stdin.flush()

    # Read scores
    assert meteor_process.stdout is not None, "INTERNAL METEOR process error"
    meteor_scores = []
    for i in range(n_candidates):
        process_out_i = meteor_process.stdout.readline().strip()
        try:
            meteor_scores_i = float(process_out_i)
        except ValueError as err:
            pylog.error(
                f"Invalid METEOR stdout. (cannot convert sentence score to float {process_out_i=} with {i=})"
            )
            raise err
        meteor_scores.append(meteor_scores_i)

    process_out = meteor_process.stdout.readline().strip()
    try:
        meteor_score = float(process_out)
    except ValueError as err:
        pylog.error(
            f"Invalid METEOR stdout. (cannot convert global score to float {process_out=})"
        )
        raise err

    meteor_process.stdin.close()
    meteor_process.kill()
    meteor_process.wait()

    dtype = torch.float64
    meteor_score = torch.as_tensor(meteor_score, dtype=dtype)
    meteor_scores = torch.as_tensor(meteor_scores, dtype=dtype)

    if return_all_scores:
        meteor_outs_corpus = {
            "meteor": meteor_score,
        }
        meteor_outs_sents = {
            "meteor": meteor_scores,
        }
        meteor_outs = meteor_outs_corpus, meteor_outs_sents
        return meteor_outs
    else:
        return meteor_score


def _encode_cand_and_refs(candidate: str, references: list[str]) -> bytes:
    # SCORE ||| reference 1 words ||| ... ||| reference N words ||| candidate words
    candidate = candidate.replace("|||", "").replace("  ", " ")
    score_line = " ||| ".join(("SCORE", " ||| ".join(references), candidate))
    encoded = "{}\n".format(score_line).encode()
    return encoded
