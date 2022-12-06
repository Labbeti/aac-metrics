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
