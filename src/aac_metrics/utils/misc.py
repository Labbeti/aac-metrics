#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess

from pathlib import Path
from subprocess import CalledProcessError
from typing import Union


def _check_java_path(java_path: Union[str, Path]) -> bool:
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
