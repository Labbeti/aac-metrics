#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import tempfile

from typing import Optional, Union


__DEFAULT_PATHS: dict[str, dict[str, Optional[str]]] = {
    "cache": {
        "user": None,
        "env": "AAC_METRICS_CACHE_PATH",
        "default": osp.expanduser(osp.join("~", ".cache")),
    },
    "java": {
        "user": None,
        "env": "AAC_METRICS_JAVA_PATH",
        "default": "java",
    },
    "tmp": {
        "user": None,
        "env": "AAC_METRICS_TMP_PATH",
        "default": tempfile.gettempdir(),
    },
}


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


def _process_cache_path(cache_path: Union[str, None]) -> str:
    return __process_path("cache", cache_path)


def _process_java_path(java_path: Union[str, None]) -> str:
    return __process_path("java", java_path)


def _process_tmp_path(tmp_path: Union[str, None]) -> str:
    return __process_path("tmp", tmp_path)


def __get_default_path(path_name: str) -> str:
    user_path = __DEFAULT_PATHS[path_name]["user"]
    if user_path is not None:
        user_path = osp.expandvars(osp.expanduser(user_path))
        return user_path

    env_var_name = __DEFAULT_PATHS[path_name]["env"]
    env_path = os.getenv(env_var_name)  # type: ignore
    if env_path is not None:
        env_path = osp.expandvars(osp.expanduser(env_path))
        return env_path

    default_path = __DEFAULT_PATHS[path_name]["default"]

    if default_path is not None:
        default_path = osp.expandvars(osp.expanduser(default_path))
        return default_path

    raise RuntimeError(
        f"Invalid default path for {path_name=}. (all default paths are None)"
    )


def __set_default_path(
    path_name: str,
    path: Optional[str],
) -> None:
    if path is not None:
        path = osp.expandvars(osp.expanduser(path))
    __DEFAULT_PATHS[path_name]["user"] = path


def __process_path(path_name: str, path: Union[str, None]) -> str:
    if path is ... or path is None:
        return __get_default_path(path_name)
    else:
        path = osp.expandvars(osp.expanduser(path))
        return path
