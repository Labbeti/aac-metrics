#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import TypeVar

T = TypeVar("T")


def flat_list(lst: list[list[T]]) -> tuple[list[T], list[int]]:
    """Return a flat version of the input list of sublists with each sublist size."""
    flatten_lst = [element for sublst in lst for element in sublst]
    sizes = [len(sents) for sents in lst]
    return flatten_lst, sizes


def unflat_list(flatten_lst: list[T], sizes: list[int]) -> list[list[T]]:
    """Unflat a list to a list of sublists of given sizes."""
    lst = []
    start = 0
    stop = 0
    for count in sizes:
        stop += count
        lst.append(flatten_lst[start:stop])
        start = stop
    return lst


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
        raise ValueError(
            f"Invalid arguments lengths. (found {len(lst)=} != {len(sizes)=})"
        )

    out_size = sum(sizes)
    out: list[T] = [None for _ in range(out_size)]  # type: ignore
    curidx = 0
    for size, elt in zip(sizes, lst):
        out[curidx : curidx + size] = [elt] * size
        curidx += size
    return out
