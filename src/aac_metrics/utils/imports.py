#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import cache
from importlib.util import find_spec


@cache
def _package_is_available(package_name: str) -> bool:
    try:
        return find_spec(package_name) is not None
    except AttributeError:
        # Python 3.6
        return False
    except (ImportError, ModuleNotFoundError):
        # Python 3.7+
        return False


_TORCHMETRICS_AVAILABLE: bool = _package_is_available("torchmetrics")
