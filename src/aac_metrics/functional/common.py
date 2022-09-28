#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess

from pathlib import Path
from subprocess import CalledProcessError
from typing import Any, Union

from torch import Tensor


MetricOutput = Union[Tensor, tuple[dict[str, Tensor], dict[str, Tensor]]]


def check_java_path(java_path: Union[str, Path]) -> bool:
    """Returns True if the java path is valid."""
    if not isinstance(java_path, (str, Path)):
        return False

    try:
        exitcode = subprocess.check_call(
            [java_path, "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (CalledProcessError, PermissionError, FileNotFoundError):
        exitcode = 1
    return exitcode == 0


def check_input(
    candidates: Any,
    mult_references: Any,
) -> None:
    cands_is_list = isinstance(candidates, list)
    cand_is_str = all(isinstance(cand, str) for cand in candidates)
    if not all((cands_is_list, cand_is_str)):
        raise ValueError("Invalid candidates type. (expected list[str])")

    mrefs_is_list = isinstance(mult_references, list)
    refs_is_list = all(isinstance(refs, list) for refs in mult_references)
    ref_is_str = all(isinstance(ref, str) for refs in mult_references for ref in refs)
    if not all((mrefs_is_list, refs_is_list, ref_is_str)):
        raise ValueError("Invalid mult_references type. (expected list[list[str]])")

    same_len = len(candidates) == len(mult_references)
    if not same_len:
        raise ValueError(
            f"Invalid number of candidates ({len(candidates)}) with the number of references ({len(mult_references)})."
        )

    at_least_1_ref_per_cand = all(len(refs) > 0 for refs in mult_references)
    if not at_least_1_ref_per_cand:
        raise ValueError(
            "Invalid number of references per candidate. (found at least 1 empty list of references)"
        )
