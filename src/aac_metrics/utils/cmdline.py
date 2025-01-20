#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional

_TRUE_VALUES = ("true", "1", "t", "yes", "y")
_FALSE_VALUES = ("false", "0", "f", "no", "n")


def _str_to_bool(s: str) -> bool:
    s = str(s).strip().lower()
    if s in _TRUE_VALUES:
        return True
    elif s in _FALSE_VALUES:
        return False
    else:
        msg = f"Invalid argument {s=}. (expected one of {_TRUE_VALUES + _FALSE_VALUES})"
        raise ValueError(msg)


def _str_to_opt_str(s: str) -> Optional[str]:
    s = str(s)
    if s.lower() == "none":
        return None
    else:
        return s
