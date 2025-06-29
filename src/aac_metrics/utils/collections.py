#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import TypeVar

T = TypeVar("T")


def duplicate_list(lst: list[T], sizes: list[int]) -> list[T]:
    """Duplicate elements elements of a list with the corresponding sizes.

    Example 1
    ----------
    >>> lst = ["a", "b", "c", "d", "e"]
    >>> sizes = [1, 0, 2, 1, 3]
    >>> duplicate_list(lst, sizes)
    ... ["a", "c", "c", "d", "e", "e", "e"]
    """
    if len(lst) != len(sizes):
        msg = f"Invalid arguments lengths. (found {len(lst)=} != {len(sizes)=})"
        raise ValueError(msg)

    out_size = sum(sizes)
    out: list[T] = [None for _ in range(out_size)]  # type: ignore
    curidx = 0
    for size, elt in zip(sizes, lst):
        out[curidx : curidx + size] = [elt] * size
        curidx += size
    return out
