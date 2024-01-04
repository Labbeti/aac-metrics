#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import os.path as osp
import tempfile

from pathlib import Path
from typing import Any, Optional, Union

import torch


pylog = logging.getLogger(__name__)


# Public functions
def get_default_cache_path() -> str:
    """Returns the default cache directory path.

    If :func:`~aac_metrics.utils.globals.set_default_cache_path` has been used before with a string argument, it will return the value given to this function.
    Else if the environment variable AAC_METRICS_CACHE_PATH has been set to a string, it will return its value.
    Else it will be equal to "~/.cache" by default.
    """
    return __get_default_value("cache")


def get_default_java_path() -> str:
    """Returns the default java executable path.

    If :func:`~aac_metrics.utils.globals.set_default_java_path` has been used before with a string argument, it will return the value given to this function.
    Else if the environment variable AAC_METRICS_JAVA_PATH has been set to a string, it will return its value.
    Else it will be equal to "java" by default.
    """
    return __get_default_value("java")


def get_default_tmp_path() -> str:
    """Returns the default temporary directory path.

    If :func:`~aac_metrics.utils.globals.set_default_tmp_path` has been used before with a string argument, it will return the value given to this function.
    Else if the environment variable AAC_METRICS_TMP_PATH has been set to a string, it will return its value.
    Else it will be equal to the value returned by :func:`~tempfile.gettempdir()` by default.
    """
    return __get_default_value("tmp")


def set_default_cache_path(cache_path: Union[str, Path, None]) -> None:
    """Override default cache directory path."""
    __set_default_value("cache", cache_path)


def set_default_java_path(java_path: Union[str, Path, None]) -> None:
    """Override default java executable path."""
    __set_default_value("java", java_path)


def set_default_tmp_path(tmp_path: Union[str, Path, None]) -> None:
    """Override default temporary directory path."""
    __set_default_value("tmp", tmp_path)


# Private functions
def _get_cache_path(cache_path: Union[str, Path, None] = None) -> str:
    return __get_value("cache", cache_path)


def _get_device(
    device: Union[str, torch.device, None] = "cuda_if_available",
) -> Optional[torch.device]:
    value_name = "device"
    process_func = __DEFAULT_GLOBALS[value_name]["process"]
    device = process_func(device)
    return device  # type: ignore


def _get_java_path(java_path: Union[str, Path, None] = None) -> str:
    return __get_value("java", java_path)


def _get_tmp_path(tmp_path: Union[str, Path, None] = None) -> str:
    return __get_value("tmp", tmp_path)


def __get_default_value(value_name: str) -> Any:
    values = __DEFAULT_GLOBALS[value_name]["values"]
    process_func = __DEFAULT_GLOBALS[value_name]["process"]
    default_val = None

    for source, value_or_env_varname in values.items():
        if source.startswith("env"):
            value = os.getenv(value_or_env_varname, default_val)
        else:
            value = value_or_env_varname

        if value != default_val:
            value = process_func(value)
            return value

    pylog.error(f"Values: {values}")
    raise RuntimeError(
        f"Invalid default value for value_name={value_name}. (all default values are None)"
    )


def __set_default_value(
    value_name: str,
    value: Any,
) -> None:
    __DEFAULT_GLOBALS[value_name]["values"]["user"] = value


def __get_value(value_name: str, value: Any = None) -> Any:
    if value is None or value is ...:
        return __get_default_value(value_name)
    else:
        process_func = __DEFAULT_GLOBALS[value_name]["process"]
        value = process_func(value)
        return value


def __process_path(value: Union[str, Path, None]) -> Union[str, None]:
    if value is None or value is ...:
        return None
    value = str(value)
    value = osp.expanduser(value)
    value = osp.expandvars(value)
    return value


def __process_device(value: Union[str, torch.device, None]) -> Optional[torch.device]:
    if value is None or value is ...:
        return None
    if value == "cuda_if_available":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(value, str):
        value = torch.device(value)
    return value


__DEFAULT_GLOBALS = {
    "cache": {
        "values": {
            "user": None,
            "env": "AAC_METRICS_CACHE_PATH",
            "package": osp.join("~", ".cache"),
        },
        "process": __process_path,
    },
    "device": {
        "values": {
            "env": "AAC_METRICS_DEVICE",
            "package": "cuda_if_available",
        },
        "process": __process_device,
    },
    "java": {
        "values": {
            "user": None,
            "env": "AAC_METRICS_JAVA_PATH",
            "package": "java",
        },
        "process": __process_path,
    },
    "tmp": {
        "values": {
            "user": None,
            "env": "AAC_METRICS_TMP_PATH",
            "package": tempfile.gettempdir(),
        },
        "process": __process_path,
    },
}
