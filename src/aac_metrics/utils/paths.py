#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import os.path as osp
import tempfile

from pathlib import Path
from typing import Union, overload


pylog = logging.getLogger(__name__)


__DEFAULT_GLOBALS: dict[str, dict[str, Union[str, None]]] = {
    "cache": {
        "user": None,
        "env": "AAC_METRICS_CACHE_PATH",
        "package": osp.expanduser(osp.join("~", ".cache")),
    },
    "java": {
        "user": None,
        "env": "AAC_METRICS_JAVA_PATH",
        "package": "java",
    },
    "tmp": {
        "user": None,
        "env": "AAC_METRICS_TMP_PATH",
        "package": tempfile.gettempdir(),
    },
}


# Public functions
def get_default_cache_path() -> str:
    """Returns the default cache directory path.

    If :func:`~aac_metrics.utils.path.set_default_cache_path` has been used before with a string argument, it will return the value given to this function.
    Else if the environment variable AAC_METRICS_CACHE_PATH has been set to a string, it will return its value.
    Else it will be equal to "~/.cache" by default.
    """
    return __get_default_value("cache")


def get_default_java_path() -> str:
    """Returns the default java executable path.

    If :func:`~aac_metrics.utils.path.set_default_java_path` has been used before with a string argument, it will return the value given to this function.
    Else if the environment variable AAC_METRICS_JAVA_PATH has been set to a string, it will return its value.
    Else it will be equal to "java" by default.
    """
    return __get_default_value("java")


def get_default_tmp_path() -> str:
    """Returns the default temporary directory path.

    If :func:`~aac_metrics.utils.path.set_default_tmp_path` has been used before with a string argument, it will return the value given to this function.
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


def _get_java_path(java_path: Union[str, Path, None] = None) -> str:
    return __get_value("java", java_path)


def _get_tmp_path(tmp_path: Union[str, Path, None] = None) -> str:
    return __get_value("tmp", tmp_path)


def __get_default_value(value_name: str) -> str:
    values = __DEFAULT_GLOBALS[value_name]

    for source, value_or_env_varname in values.items():
        if value_or_env_varname is None:
            continue

        if source.startswith("env"):
            path = os.getenv(value_or_env_varname, None)
        else:
            path = value_or_env_varname

        if path is not None:
            path = __process_value(path)
            return path

    pylog.error(f"Paths values: {values}")
    raise RuntimeError(
        f"Invalid default path for {value_name=}. (all default paths are None)"
    )


def __set_default_value(
    value_name: str,
    value: Union[str, Path, None],
) -> None:
    value = __process_value(value)
    __DEFAULT_GLOBALS[value_name]["user"] = value


def __get_value(value_name: str, value: Union[str, Path, None] = None) -> str:
    if value is ... or value is None:
        return __get_default_value(value_name)
    else:
        value = __process_value(value)
        return value


@overload
def __process_value(value: None) -> None:
    ...


@overload
def __process_value(value: Union[str, Path]) -> str:
    ...


def __process_value(value: Union[str, Path, None]) -> Union[str, None]:
    value = str(value)
    value = osp.expanduser(value)
    value = osp.expandvars(value)
    return value
