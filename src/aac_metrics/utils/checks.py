#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import re
import subprocess

from functools import cache
from pathlib import Path
from subprocess import CalledProcessError
from typing import Any, Union


pylog = logging.getLogger(__name__)

VERSION_PATTERN = r"(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+).*"
MIN_JAVA_MAJOR_VERSION = 8
MAX_JAVA_MAJOR_VERSION = 13


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
    version = _get_java_version(str(java_path))
    valid = _check_java_version(version, MIN_JAVA_MAJOR_VERSION, MAX_JAVA_MAJOR_VERSION)
    if not valid:
        pylog.error(
            f"Using Java version {version} is not officially supported by aac-metrics package and will not work for METEOR and SPICE metrics."
            f"(expected major version in range [{MIN_JAVA_MAJOR_VERSION}, {MAX_JAVA_MAJOR_VERSION}])"
        )
    return valid


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


def _get_java_version(java_path: str) -> str:
    """Returns True if the java path is valid."""
    if not isinstance(java_path, str):
        raise TypeError(f"Invalid argument type {type(java_path)=}. (expected str)")

    output = "INVALID"
    try:
        output = subprocess.check_output(
            [java_path, "-version"],
            stderr=subprocess.STDOUT,
        )
        output = output.decode().strip()
        version = output.split(" ")[2][1:-1]

    except (
        CalledProcessError,
        PermissionError,
        FileNotFoundError,
    ) as err:
        raise ValueError(f"Invalid java path. (from {java_path=} and found {err=})")

    except IndexError as err:
        raise ValueError(
            f"Invalid java version. (from {java_path=} and found {output=} and {err=})"
        )

    return version


def _check_java_version(version: str, min_major: int, max_major: int) -> bool:
    result = re.match(VERSION_PATTERN, version)
    if result is None:
        raise ValueError(
            f"Invalid Java version {version=}. (expected version with pattern={VERSION_PATTERN})"
        )

    major_version = int(result["major"])
    minor_version = int(result["minor"])

    if (
        major_version == 1 and minor_version <= 8
    ):  # java <= 8 use versioning "1.MAJOR.MINOR" and > 8 use "MAJOR.MINOR.PATCH"
        major_version = minor_version

    return min_major <= major_version <= max_major


@cache
def _warn_once(msg: str) -> None:
    pylog.warning(msg)
