#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import json
import logging
import math
import os
import os.path as osp
import platform
import shutil
import subprocess
import tempfile
import time

from pathlib import Path
from subprocess import CalledProcessError
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Optional, Union

import numpy as np
import torch

from torch import Tensor

from aac_metrics.utils.checks import check_java_path, check_metric_inputs
from aac_metrics.utils.globals import (
    _get_cache_path,
    _get_java_path,
    _get_tmp_path,
)


pylog = logging.getLogger(__name__)


DNAME_SPICE_CACHE = osp.join("aac-metrics", "spice")
DNAME_SPICE_LOCAL_CACHE = osp.join(DNAME_SPICE_CACHE, "cache")
FNAME_SPICE_JAR = osp.join(DNAME_SPICE_CACHE, "spice-1.0.jar")


def spice(
    candidates: list[str],
    mult_references: list[list[str]],
    return_all_scores: bool = True,
    cache_path: Union[str, Path, None] = None,
    java_path: Union[str, Path, None] = None,
    tmp_path: Union[str, Path, None] = None,
    n_threads: Optional[int] = None,
    java_max_memory: str = "8G",
    timeout: Union[None, int, Iterable[int]] = None,
    separate_cache_dir: bool = True,
    use_shell: Optional[bool] = None,
    verbose: int = 0,
) -> Union[tuple[dict[str, Tensor], dict[str, Tensor]], Tensor]:
    """Semantic Propositional Image Caption Evaluation function.

    - Paper: https://arxiv.org/pdf/1607.08822.pdf

    :param candidates: The list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target.
    :param return_all_scores: If True, returns a tuple containing the globals and locals scores.
        Otherwise returns a scalar tensor containing the main global score.
        defaults to True.
    :param cache_path: The path to the external code directory. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_cache_path`.
    :param java_path: The path to the java executable. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_java_path`.
    :param tmp_path: Temporary directory path. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_tmp_path`.
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
    :param use_shell: Optional argument to force use os-specific shell for the java subprogram.
        If None, it will use shell only on Windows OS.
        defaults to None.
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple of globals and locals scores or a scalar tensor with the main global score.
    """
    check_metric_inputs(candidates, mult_references)

    cache_path = _get_cache_path(cache_path)
    java_path = _get_java_path(java_path)
    tmp_path = _get_tmp_path(tmp_path)

    # Sometimes the java program can freeze, so timeout has been added to avoid using job time.
    if timeout is None or isinstance(timeout, (int, float)):
        timeout_lst = [timeout]
    else:
        timeout_lst = list(timeout)

    timeout_lst: list[Optional[int]]
    if len(timeout_lst) == 0:
        raise ValueError(
            f"Invalid argument {timeout_lst=}. (cannot call SPICE with empty number of timeouts)"
        )

    spice_fpath = osp.join(cache_path, FNAME_SPICE_JAR)

    if use_shell is None:
        use_shell = platform.system() == "Windows"

    if __debug__:
        check_spice_install(cache_path)

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
        spice_cache = osp.join(cache_path, DNAME_SPICE_LOCAL_CACHE)
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

    json_kwds: dict[str, Any] = dict(
        mode="w",
        delete=False,
        dir=tmp_path,
        suffix=".json",
    )
    in_file = NamedTemporaryFile(prefix="spice_inputs_", **json_kwds)
    json.dump(input_data, in_file, indent=2)
    in_file.close()

    out_file = NamedTemporaryFile(prefix="spice_outputs_", **json_kwds)
    out_file.close()

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

    fpaths = [
        java_path,
        spice_fpath,
        in_file.name,
        spice_cache,
        out_file.name,
    ]

    for i, timeout_i in enumerate(timeout_lst):
        success = __run_spice(
            i=i,
            timeout_i=timeout_i,
            timeout_lst=timeout_lst,
            spice_cmd=spice_cmd,
            tmp_path=tmp_path,
            out_path=out_file.name,
            paths=fpaths,
            use_shell=use_shell,
            verbose=verbose,
        )
        if success:
            break

    if verbose >= 2:
        pylog.debug("SPICE java code finished.")

    # Read and process results
    with open(out_file.name, "r") as data_file:
        results = json.load(data_file)
    os.remove(in_file.name)
    os.remove(out_file.name)

    if separate_cache_dir:
        shutil.rmtree(spice_cache)

    spice_scores = []
    for item in results:
        # item keys: "image_id", "scores"
        spice_scores_i = __float_convert(item["scores"]["All"]["f"])
        spice_scores.append(spice_scores_i)

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


def check_spice_install(cache_path: str) -> None:
    """Check if SPICE is installed in cache directory.

    Raises FileNotFoundError or NotADirectoryError exception if something is missing.
    """
    spice_fpath = osp.join(cache_path, FNAME_SPICE_JAR)
    if not osp.isfile(spice_fpath):
        raise FileNotFoundError(
            f"Cannot find JAR file '{spice_fpath}' for SPICE metric. Maybe run 'aac-metrics-download' or specify another 'cache_path' directory."
        )

    local_cache_dpath = osp.join(cache_path, DNAME_SPICE_CACHE, "cache")
    if not osp.isdir(local_cache_dpath):
        raise NotADirectoryError(
            f"Cannot find cache local directory '{local_cache_dpath}' for SPICE metric. Maybe run 'aac-metrics-download' or specify another 'cache_path' directory."
        )

    lib_dpath = osp.join(cache_path, DNAME_SPICE_CACHE, "lib")
    if not osp.isdir(lib_dpath):
        raise NotADirectoryError(
            f"Cannot find lib directory '{lib_dpath}' for SPICE metric. Maybe run 'aac-metrics-download' or specify another 'cache_path' directory."
        )

    expected_jar_in_lib = [
        "ejml-0.23.jar",
        "fst-2.47.jar",
        "guava-19.0.jar",
        "hamcrest-core-1.3.jar",
        "jackson-core-2.5.3.jar",
        "javassist-3.19.0-GA.jar",
        "json-simple-1.1.1.jar",
        "junit-4.12.jar",
        "lmdbjni-0.4.6.jar",
        "lmdbjni-linux64-0.4.6.jar",
        "lmdbjni-osx64-0.4.6.jar",
        "lmdbjni-win64-0.4.6.jar",
        "Meteor-1.5.jar",
        "objenesis-2.4.jar",
        "SceneGraphParser-1.0.jar",
        "slf4j-api-1.7.12.jar",
        "slf4j-simple-1.7.21.jar",
        "stanford-corenlp-3.6.0.jar",
        "stanford-corenlp-3.6.0-models.jar",
    ]
    names = os.listdir(lib_dpath)
    files_not_found = []
    for fname in expected_jar_in_lib:
        if fname not in names:
            files_not_found.append(fname)
    if len(files_not_found) > 0:
        raise FileNotFoundError(
            f"Missing {len(files_not_found)} files in SPICE lib directory. (missing {', '.join(files_not_found)})"
        )


def __run_spice(
    i: int,
    timeout_i: Optional[int],
    timeout_lst: list[Optional[int]],
    spice_cmd: list[str],
    tmp_path: str,
    out_path: str,
    paths: list[str],
    use_shell: bool,
    verbose: int,
) -> bool:
    success = False
    txt_kwds: dict[str, Any] = dict(
        mode="w",
        delete=False,
        dir=tmp_path,
        suffix=".txt",
    )

    if verbose >= 3:
        stdout = None
        stderr = None
    else:
        stdout = NamedTemporaryFile(
            prefix="spice_stdout_",
            **txt_kwds,
        )
        stderr = NamedTemporaryFile(
            prefix="spice_stderr_",
            **txt_kwds,
        )

    if verbose >= 2:
        pylog.debug(f"Run SPICE java code with: {' '.join(spice_cmd)} and {use_shell=}")

    try:
        subprocess.check_call(
            spice_cmd,
            stdout=stdout,
            stderr=stderr,
            timeout=timeout_i,
            shell=use_shell,
        )
        if stdout is not None:
            stdout.close()
            os.remove(stdout.name)
        if stderr is not None:
            stderr.close()
            os.remove(stderr.name)

        success = True

    except subprocess.TimeoutExpired as err:
        pylog.warning(
            f"Timeout SPICE java program with {timeout_i=}s (nb timeouts done={i+1}/{len(timeout_lst)})."
        )

        if i < len(timeout_lst) - 1:
            # Clear out files
            open(out_path, "w").close()
            if stdout is not None:
                stdout.close()
                open(stdout.name, "w").close()
            if stderr is not None:
                stderr.close()
                open(stderr.name, "w").close()
            time.sleep(1.0)
        else:
            raise err

    except (CalledProcessError, PermissionError) as err:
        pylog.error("Invalid SPICE call.")
        pylog.error(f"Full command: '{' '.join(spice_cmd)}'")
        pylog.error(f"Error: {err}")

        paths = copy.copy(paths)
        if stdout is not None:
            stdout.close()
            paths.append(stdout.name)
        if stderr is not None:
            stderr.close()
            paths.append(stderr.name)

        for path in paths:
            rights = __get_access_rights(path)
            pylog.error(f"{path} :\t {rights}")

        if (
            stdout is not None
            and stderr is not None
            and osp.isfile(stdout.name)
            and osp.isfile(stderr.name)
        ):
            pylog.error(
                f"For more information, see temp files '{stdout.name}' and '{stderr.name}'."
            )

            for path in (stdout.name, stderr.name):
                try:
                    with open(path, "r") as file:
                        lines = file.readlines()
                    content = "\n".join(lines)
                    pylog.error(f"Content of '{path}':\n{content}")
                except PermissionError as err2:
                    pylog.warning(f"Cannot open file '{path}'. ({err2})")
        else:
            pylog.info(f"Note: No temp file recorded. (found {stdout=} and {stderr=})")
        raise err

    return success


def __get_access_rights(path: str) -> str:
    info = {"t": "-", "r": "-", "w": "-", "x": "-"}
    if osp.islink(path):
        info["t"] = "l"
    elif osp.isfile(path):
        info["t"] = "f"
    elif osp.isdir(path):
        info["t"] = "d"

    if os.access(path, os.R_OK):
        info["r"] = "r"
    if os.access(path, os.W_OK):
        info["w"] = "w"
    if os.access(path, os.X_OK):
        info["x"] = "x"

    rights = "".join(info.values())
    return rights


def __float_convert(obj: Any) -> float:
    try:
        return float(obj)
    except (ValueError, TypeError):
        return math.nan
