#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import math
import os
import os.path as osp
import subprocess

from subprocess import CalledProcessError
from tempfile import NamedTemporaryFile
from typing import Any, Optional, Union

import numpy as np
import torch

from torch import Tensor

from aac_metrics.utils.misc import _check_java_path


logger = logging.getLogger(__name__)


SPICE_JAR_FNAME = osp.join("spice", "spice-1.0.jar")
CACHE_DNAME = "spice_cache"


def coco_spice(
    candidates: list[str],
    mult_references: list[list[str]],
    return_all_scores: bool = True,
    cache_path: str = "$HOME/aac-metrics-cache",
    java_path: str = "java",
    tmp_path: str = "/tmp",
    n_threads: Optional[int] = None,
    java_max_memory: str = "8G",
    verbose: int = 0,
) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
    """Semantic Propositional Image Caption Evaluation function.

    Paper: https://arxiv.org/pdf/1607.08822.pdf

    :param candidates: The list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target.
    :param return_all_scores: If True, returns a tuple containing the globals and locals scores.
        Otherwise returns a scalar tensor containing the main global score.
        defaults to True.
    :param cache_path: The path to the external code directory. defaults to "$HOME/aac-metrics-cache".
    :param java_path: The path to the java executable. defaults to "java".
    :param tmp_path: Temporary directory path. defaults to "/tmp".
    :param java_max_memory: The maximal java memory used. defaults to "8G".
    :param n_threads: Number of threads used to compute SPICE.
        None value will use the default value of the java program.
        defaults to None.
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple of globals and locals scores or a scalar tensor with the main global score.
    """

    cache_path = osp.expandvars(cache_path)
    java_path = osp.expandvars(java_path)
    tmp_path = osp.expandvars(tmp_path)

    if not _check_java_path(java_path):
        raise ValueError(
            f"Cannot find java executable with {java_path=} for compute SPICE metric score."
        )

    spice_fpath = osp.join(cache_path, SPICE_JAR_FNAME)
    if not osp.isfile(spice_fpath):
        raise FileNotFoundError(
            f"Cannot find JAR file '{SPICE_JAR_FNAME}' in directory '{cache_path}' for SPICE metric. Maybe run 'aac-metrics-download' before or specify a 'cache_path' directory."
        )

    if len(candidates) != len(mult_references):
        raise ValueError(
            f"Invalid number of candidates and references. (found {len(candidates)=} != {len(mult_references)=})"
        )

    os.makedirs(cache_path, exist_ok=True)

    if verbose >= 2:
        logger.debug(f"Use cache directory {cache_path}.")
        logger.debug(f"Computing SPICE with JAR file {spice_fpath}...")

    input_data = [
        {
            "image_id": i,
            "test": cand,
            "refs": refs,
        }
        for i, (cand, refs) in enumerate(zip(candidates, mult_references))
    ]

    in_file = NamedTemporaryFile(
        mode="w", delete=False, dir=tmp_path, suffix=".json", prefix="in_"
    )
    json.dump(input_data, in_file, indent=2)
    in_file.close()

    out_file = NamedTemporaryFile(
        mode="w", delete=False, dir=tmp_path, suffix=".json", prefix="out_"
    )
    out_file.close()

    if verbose >= 3:
        stdout = None
        stderr = None
    else:
        stdout = subprocess.DEVNULL
        stderr = subprocess.DEVNULL

    spice_cmd = [
        java_path,
        "-jar",
        f"-Xmx{java_max_memory}",
        spice_fpath,
        in_file.name,
        "-cache",
        cache_path,
        "-out",
        out_file.name,
        "-subset",
    ]
    if n_threads is not None:
        spice_cmd += ["-threads", str(n_threads)]

    if verbose >= 2:
        logger.debug(f"Run SPICE java code with: {' '.join(spice_cmd)}")

    try:
        subprocess.check_call(
            spice_cmd,
            stdout=stdout,
            stderr=stderr,
        )
    except (CalledProcessError, PermissionError) as err:
        logger.error(
            f"Invalid SPICE call. (full_command='{' '.join(spice_cmd)}', {err=})"
        )
        raise err

    if verbose >= 2:
        logger.debug("SPICE java code finished.")

    # Read and process results
    with open(out_file.name, "r") as data_file:
        results = json.load(data_file)
    os.remove(in_file.name)
    os.remove(out_file.name)

    imgId_to_scores = {}
    spice_scores = []
    for item in results:
        imgId_to_scores[item["image_id"]] = item["scores"]
        spice_scores.append(__float_convert(item["scores"]["All"]["f"]))

    spice_scores = np.array(spice_scores)
    # Note: use numpy to compute mean because np.mean and torch.mean can give very small differences
    spice_score = spice_scores.mean()

    dtype = torch.float64
    spice_scores = torch.from_numpy(spice_scores)
    spice_score = torch.as_tensor(spice_score, dtype=dtype)

    if return_all_scores:
        global_scores = {
            "spice": spice_score,
        }
        local_scores = {
            "spice": spice_scores,
        }
        return global_scores, local_scores
    else:
        return spice_score


def __float_convert(obj: Any) -> float:
    try:
        return float(obj)
    except (ValueError, TypeError):
        return math.nan
