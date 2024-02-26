#!/usr/bin/env python
# -*- coding: utf-8 -*-

import platform
import sys

from pathlib import Path
from typing import Dict

import torch
import yaml

import aac_metrics

from aac_metrics.utils.checks import _get_java_version
from aac_metrics.utils.globals import (
    get_default_cache_path,
    get_default_java_path,
    get_default_tmp_path,
)


def get_package_repository_path() -> str:
    """Return the absolute path where the source code of this package is installed."""
    return str(Path(__file__).parent.parent.parent)


def get_java_version() -> str:
    try:
        java_version = _get_java_version(get_default_java_path())
        return java_version
    except ValueError:
        return "UNKNOWN"


def get_install_info() -> Dict[str, str]:
    """Return a dictionary containing the version python, the os name, the architecture name and the versions of the following packages: aac_datasets, torch, torchaudio."""

    return {
        "aac_metrics": aac_metrics.__version__,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "os": platform.system(),
        "architecture": platform.architecture()[0],
        "torch": str(torch.__version__),
        "package_path": get_package_repository_path(),
        "cache_path": get_default_cache_path(),
        "java_path": get_default_java_path(),
        "tmp_path": get_default_tmp_path(),
        "java_version": get_java_version(),
    }


def print_install_info() -> None:
    """Show main packages versions."""
    install_info = get_install_info()
    print(yaml.dump(install_info, sort_keys=False))


if __name__ == "__main__":
    print_install_info()
