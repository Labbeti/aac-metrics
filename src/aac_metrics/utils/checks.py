#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import subprocess

from functools import cache
from pathlib import Path
from subprocess import CalledProcessError
from typing import Any, Union


pylog = logging.getLogger(__name__)

MIN_JAVA_MAJOR_VERSION = 8
MAX_JAVA_MAJOR_VERSION = 11


def check_metric_inputs(
    candidates: Any,
    mult_references: Any,
) -> None:
    """Raises ValueError if candidates and mult_references does not have a valid type and size."""
    if not is_mono_sents(candidates):
        raise ValueError("Invalid candidates type. (expected list[str])")

    if not is_mult_sents(mult_references):
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


def check_java_path(java_path: Union[str, Path]) -> bool:
    """Returns True if the java path is valid."""
    if not isinstance(java_path, (str, Path)):
        return False

    output = ""
    try:
        output = subprocess.check_output(
            [str(java_path), "--version"],
        )
        output = output.decode().strip()
        version = output.split("\n")[0]
        major_version = int(version.split(" ")[1].split(".")[0])

    except (
        CalledProcessError,
        PermissionError,
        FileNotFoundError,
    ) as err:
        pylog.error(f"Invalid java path. (from {java_path=} and found error={err})")
        return False

    except (
        IndexError,
        ValueError,
    ) as err:
        pylog.error(f"Invalid java version. (found {output=} and {err=})")
        return False

    if not (MIN_JAVA_MAJOR_VERSION <= major_version <= MAX_JAVA_MAJOR_VERSION):
        pylog.error(
            f"Using Java version {version} is not officially supported by aac-metrics package and could not work for METEOR and SPICE metrics."
            f"(found {major_version=} but expected in [{MIN_JAVA_MAJOR_VERSION}, {MAX_JAVA_MAJOR_VERSION}])"
        )
        return False

    return True


def is_mono_sents(sents: Any) -> bool:
    """Returns True if input is list[str]."""
    return isinstance(sents, list) and all(isinstance(sent, str) for sent in sents)


def is_mult_sents(mult_sents: Any) -> bool:
    """Returns True if input is list[list[str]]."""
    return (
        isinstance(mult_sents, list)
        and all(isinstance(sents, list) for sents in mult_sents)
        and all(isinstance(sent, str) for sents in mult_sents for sent in sents)
    )


@cache
def _warn_once(msg: str) -> None:
    pylog.warning(msg)
