#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import logging
import math
import os
import os.path as osp
import shutil
import subprocess
import tempfile
import time

from subprocess import CalledProcessError
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Optional, Union

import numpy as np
import torch

from torch import Tensor

from aac_metrics.utils.checks import check_java_path


pylog = logging.getLogger(__name__)


DNAME_SPICE_CACHE = osp.join("aac-metrics", "spice", "cache")
FNAME_SPICE_JAR = osp.join("aac-metrics", "spice", "spice-1.0.jar")


def spice(
    candidates: list[str],
    mult_references: list[list[str]],
    return_all_scores: bool = True,
    cache_path: str = "$HOME/.cache",
    java_path: str = "java",
    tmp_path: str = "/tmp",
    n_threads: Optional[int] = None,
    java_max_memory: str = "8G",
    timeout: Union[None, int, Iterable[int]] = None,
    separate_cache_dir: bool = True,
    verbose: int = 0,
) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
    """Semantic Propositional Image Caption Evaluation function.

    - Paper: https://arxiv.org/pdf/1607.08822.pdf

    :param candidates: The list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target.
    :param return_all_scores: If True, returns a tuple containing the globals and locals scores.
        Otherwise returns a scalar tensor containing the main global score.
        defaults to True.
    :param cache_path: The path to the external code directory. defaults to "$HOME/.cache".
    :param java_path: The path to the java executable. defaults to "java".
    :param tmp_path: Temporary directory path. defaults to "/tmp".
    :param n_threads: Number of threads used to compute SPICE.
        None value will use the default value of the java program.
        defaults to None.
    :param java_max_memory: The maximal java memory used. defaults to "8G".
    :param timeout: The number of seconds before killing the java subprogram.
        If a list is given, it will restart the program if the i-th timeout is reached.
        If None, no timeout will be used.
        defaults to None.
    :param separate_cache_dir: If True, the SPICE cache files will be stored into in a new temporary directory.
        This removes potential freezes when multiple instances of SPICE are running in the same cache dir.
        defaults to True.
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple of globals and locals scores or a scalar tensor with the main global score.
    """

    cache_path = osp.expandvars(cache_path)
    java_path = osp.expandvars(java_path)
    tmp_path = osp.expandvars(tmp_path)

    spice_fpath = osp.join(cache_path, FNAME_SPICE_JAR)

    if __debug__:
        if not osp.isfile(spice_fpath):
            raise FileNotFoundError(
                f"Cannot find JAR file '{spice_fpath}' for SPICE metric. Maybe run 'aac-metrics-download' or specify another 'cache_path' directory."
            )
        if not check_java_path(java_path):
            raise RuntimeError(
                f"Invalid Java executable to compute SPICE score. ({java_path})"
            )

    if len(candidates) != len(mult_references):
        raise ValueError(
            f"Invalid number of candidates and references. (found {len(candidates)=} != {len(mult_references)=})"
        )

    if separate_cache_dir:
        spice_cache = tempfile.mkdtemp(dir=tmp_path)
    else:
        spice_cache = osp.join(cache_path, DNAME_SPICE_CACHE)
    del cache_path

    if verbose >= 2:
        pylog.debug(f"Use cache directory {spice_cache}.")
        pylog.debug(f"Computing SPICE with JAR file {spice_fpath}...")

    input_data = [
        {
            "image_id": i,
            "test": cand,
            "refs": refs,
        }
        for i, (cand, refs) in enumerate(zip(candidates, mult_references))
    ]

    in_file = NamedTemporaryFile(
        mode="w", delete=False, dir=tmp_path, prefix="spice_inputs_", suffix=".json"
    )
    json.dump(input_data, in_file, indent=2)
    in_file.close()

    # Sometimes the java program can freeze, so timeout has been added to avoid using job time.
    if timeout is None or isinstance(timeout, (int, float)):
        timeout_lst = [timeout]
    else:
        timeout_lst = list(timeout)

    out_file = NamedTemporaryFile(
        mode="w", delete=False, dir=tmp_path, prefix="spice_outputs_", suffix=".json"
    )
    out_file.close()

    for i, timeout_i in enumerate(timeout_lst):
        if verbose >= 3:
            stdout = None
            stderr = None
        else:
            stdout = NamedTemporaryFile(
                mode="w",
                delete=True,
                dir=tmp_path,
                prefix="spice_stdout_",
                suffix=".txt",
            )
            stderr = NamedTemporaryFile(
                mode="w",
                delete=True,
                dir=tmp_path,
                prefix="spice_stderr_",
                suffix=".txt",
            )

        spice_cmd = [
            java_path,
            "-jar",
            f"-Xmx{java_max_memory}",
            spice_fpath,
            in_file.name,
            "-cache",
            spice_cache,
            "-out",
            out_file.name,
            "-subset",
        ]
        if n_threads is not None:
            spice_cmd += ["-threads", str(n_threads)]

        if verbose >= 2:
            pylog.debug(f"Run SPICE java code with: {' '.join(spice_cmd)}")

        try:
            subprocess.check_call(
                spice_cmd,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout_i,
            )
            if stdout is not None:
                stdout.close()
            if stderr is not None:
                stderr.close()
            break

        except subprocess.TimeoutExpired as err:
            pylog.warning(
                f"Timeout SPICE java program with {timeout_i=}s (nb timeouts done={i+1}/{len(timeout_lst)})."
            )

            if i < len(timeout_lst) - 1:
                # Clear out files
                open(out_file.name, "w").close()
                if stdout is not None:
                    stdout.close()
                if stderr is not None:
                    stderr.close()
                time.sleep(1.0)
            else:
                raise err

        except (CalledProcessError, PermissionError) as err:
            pylog.error("Invalid SPICE call.")
            pylog.error(f"Full command: '{' '.join(spice_cmd)}'")
            if (
                stdout is not None
                and stderr is not None
                and osp.isfile(stdout.name)
                and osp.isfile(stderr.name)
            ):
                stdout_crashlog = stdout.name.replace(
                    "spice_stdout", "CRASH_spice_stdout"
                )
                stderr_crashlog = stderr.name.replace(
                    "spice_stderr", "CRASH_spice_stderr"
                )
                shutil.copy(stdout.name, stdout_crashlog)
                shutil.copy(stderr.name, stderr_crashlog)
                pylog.error(
                    f"For more information, see temp files '{stdout_crashlog}' and '{stderr_crashlog}'."
                )
            else:
                pylog.info(
                    f"Note: No temp file recorded. (found {stdout=} and {stderr=})"
                )
            raise err

    if verbose >= 2:
        pylog.debug("SPICE java code finished.")

    # Read and process results
    with open(out_file.name, "r") as data_file:
        results = json.load(data_file)
    os.remove(in_file.name)
    os.remove(out_file.name)

    if separate_cache_dir:
        shutil.rmtree(spice_cache)

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
        spice_outs_corpus = {
            "spice": spice_score,
        }
        spice_outs_sents = {
            "spice": spice_scores,
        }
        return spice_outs_corpus, spice_outs_sents
    else:
        return spice_score


def __float_convert(obj: Any) -> float:
    try:
        return float(obj)
    except (ValueError, TypeError):
        return math.nan
