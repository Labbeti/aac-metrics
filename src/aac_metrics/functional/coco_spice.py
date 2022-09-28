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

from aac_metrics.functional.common import check_java_path


logger = logging.getLogger(__name__)


SPICE_JAR_FNAME = osp.join("spice", "spice-1.0.jar")
CACHE_DNAME = "spice_cache"


def coco_spice(
    candidates: list[str],
    mult_references: list[list[str]],
    return_all_scores: bool = True,
    java_path: str = "java",
    tmp_path: str = "/tmp",
    cache_path: str = "$HOME/aac-metrics-cache",
    n_threads: Optional[int] = None,
    java_max_memory: str = "8G",
    verbose: int = 0,
) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Tensor]]]:
    """Compute SPICE metric score."""

    java_path = osp.expandvars(java_path)
    tmp_path = osp.expandvars(tmp_path)
    cache_path = osp.expandvars(cache_path)

    if not check_java_path(java_path):
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

    if verbose >= 2:
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
        spice_scores.append(_float_convert(item["scores"]["All"]["f"]))

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


def _float_convert(obj: Any) -> float:
    try:
        return float(obj)
    except (ValueError, TypeError):
        return math.nan
