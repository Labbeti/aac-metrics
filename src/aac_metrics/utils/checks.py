#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import subprocess
from pathlib import Path
from subprocess import CalledProcessError
from typing import Any, Union

from packaging.version import Version
from typing_extensions import TypeGuard

pylog = logging.getLogger(__name__)

MIN_JAVA_MAJOR_VERSION = 8
MAX_JAVA_MAJOR_VERSION = 13


def check_metric_inputs(
    candidates: Any,
    mult_references: Any,
    min_length: int = 0,
) -> None:
    """Raises ValueError if candidates and mult_references does not have a valid type and size."""

    msgs = []
    if not is_mono_sents(candidates):
        if isinstance(candidates, list) and len(candidates) > 0:
            clsname = (
                f"{candidates.__class__.__name__}[{candidates[0].__class__.__name__}]"
            )
        else:
            clsname = candidates.__class__.__name__

        msg = f"Invalid candidates type. (expected list[str], found {clsname})"
        msgs.append(msg)

    if not is_mult_sents(mult_references):
        clsname = mult_references.__class__.__name__
        msg = (
            f"Invalid mult_references type. (expected list[list[str]], found {clsname})"
        )
        msgs.append(msg)

    if len(msgs) > 0:
        raise ValueError("\n".join(msgs))

    same_len = len(candidates) == len(mult_references)
    if not same_len:
        msg = f"Invalid number of candidates ({len(candidates)}) with the number of references ({len(mult_references)})."
        raise ValueError(msg)

    at_least_1_ref_per_cand = all(len(refs) > 0 for refs in mult_references)
    if not at_least_1_ref_per_cand:
        msg = "Invalid number of references per candidate. (found at least 1 empty list of references)"
        raise ValueError(msg)

    if len(candidates) < min_length:
        msg = f"Invalid number of sentences in candidates and references. (expected at least {min_length} sentences but found {len(candidates)=})"
        raise ValueError(msg)


def check_java_path(java_path: Union[str, Path]) -> bool:
    version = _get_java_version(str(java_path))
    valid = _check_java_version(version, MIN_JAVA_MAJOR_VERSION, MAX_JAVA_MAJOR_VERSION)
    if not valid:
        pylog.error(
            f"Using Java version {version} is not officially supported by aac-metrics package and will not work for METEOR and SPICE metrics."
            f"(expected major version in range [{MIN_JAVA_MAJOR_VERSION}, {MAX_JAVA_MAJOR_VERSION}])"
        )
    return valid


def is_mono_sents(sents: Any) -> TypeGuard[list[str]]:
    """Returns True if input is list[str] containing sentences."""
    valid = isinstance(sents, list) and all(isinstance(sent, str) for sent in sents)
    return valid


def is_mult_sents(mult_sents: Any) -> TypeGuard[list[list[str]]]:
    """Returns True if input is list[list[str]] containing multiple sentences."""
    valid = (
        isinstance(mult_sents, list)
        and all(isinstance(sents, list) for sents in mult_sents)
        and all(isinstance(sent, str) for sents in mult_sents for sent in sents)
    )
    return valid


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


def _check_java_version(version_str: str, min_major: int, max_major: int) -> bool:
    version = Version(version_str)

    if version.major == 1 and version.minor <= 8:
        # java <= 8 use versioning "1.MAJOR.MINOR" and > 8 use "MAJOR.MINOR.MICRO"
        version_str = ".".join(map(str, (version.minor, version.micro)))
        version = Version(version_str)

    return Version(f"{min_major}") <= version < Version(f"{max_major + 1}")
