#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import os.path as osp
import tempfile

from typing import Optional, Union


pylog = logging.getLogger(__name__)


__DEFAULT_PATHS: dict[str, dict[str, Optional[str]]] = {
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
    return __get_default_path("cache")


def get_default_java_path() -> str:
    """Returns the default java executable path.

    If :func:`~aac_metrics.utils.path.set_default_java_path` has been used before with a string argument, it will return the value given to this function.
    Else if the environment variable AAC_METRICS_JAVA_PATH has been set to a string, it will return its value.
    Else it will be equal to "java" by default.
    """
    return __get_default_path("java")


def get_default_tmp_path() -> str:
    """Returns the default temporary directory path.

    If :func:`~aac_metrics.utils.path.set_default_tmp_path` has been used before with a string argument, it will return the value given to this function.
    Else if the environment variable AAC_METRICS_TMP_PATH has been set to a string, it will return its value.
    Else it will be equal to the value returned by :func:`~tempfile.gettempdir()` by default.
    """
    return __get_default_path("tmp")


def set_default_cache_path(cache_path: Optional[str]) -> None:
    """Override default cache directory path."""
    __set_default_path("cache", cache_path)


def set_default_java_path(java_path: Optional[str]) -> None:
    """Override default java executable path."""
    __set_default_path("java", java_path)


def set_default_tmp_path(tmp_path: Optional[str]) -> None:
    """Override default temporary directory path."""
    __set_default_path("tmp", tmp_path)


# Private functions
def _get_cache_path(cache_path: Union[str, None] = ...) -> str:
    return __get_path("cache", cache_path)


def _get_java_path(java_path: Union[str, None] = ...) -> str:
    return __get_path("java", java_path)


def _get_tmp_path(tmp_path: Union[str, None] = ...) -> str:
    return __get_path("tmp", tmp_path)


def __get_default_path(path_name: str) -> str:
    paths = __DEFAULT_PATHS[path_name]

    for name, path_or_var in paths.items():
        if path_or_var is None:
            continue

        if name.startswith("env"):
            path = os.getenv(path_or_var, None)
        else:
            path = path_or_var

        if path is not None:
            path = __process_path(path)
            return path

    pylog.error(f"Paths values: {paths}")
    raise RuntimeError(
        f"Invalid default path for {path_name=}. (all default paths are None)"
    )


def __set_default_path(
    path_name: str,
    path: Optional[str],
) -> None:
    if path is not ... and path is not None:
        path = __process_path(path)
    __DEFAULT_PATHS[path_name]["user"] = path


def __get_path(path_name: str, path: Union[str, None] = ...) -> str:
    if path is ... or path is None:
        return __get_default_path(path_name)
    else:
        path = __process_path(path)
        return path


def __process_path(path: str) -> str:
    path = osp.expanduser(path)
    path = osp.expandvars(path)
    return path
