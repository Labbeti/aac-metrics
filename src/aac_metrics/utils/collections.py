#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Sequence,
    TypeVar,
    Union,
    overload,
)

from typing_extensions import Literal

K = TypeVar("K")
T = TypeVar("T")
V = TypeVar("V")
W = TypeVar("W")

KEY_MODES = ("same", "intersect", "union")
KeyMode = Literal["intersect", "same", "union"]


def flat_list_of_list(lst: list[list[T]]) -> tuple[list[T], list[int]]:
    """Return a flat version of the input list of sublists with each sublist size."""
    flatten_lst = [element for sublst in lst for element in sublst]
    sizes = [len(sents) for sents in lst]
    return flatten_lst, sizes


def unflat_list_of_list(flatten_lst: list[T], sizes: list[int]) -> list[list[T]]:
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
        msg = f"Invalid arguments lengths. (found {len(lst)=} != {len(sizes)=})"
        raise ValueError(msg)

    out_size = sum(sizes)
    out: list[T] = [None for _ in range(out_size)]  # type: ignore
    curidx = 0
    for size, elt in zip(sizes, lst):
        out[curidx : curidx + size] = [elt] * size
        curidx += size
    return out


@overload
def list_dict_to_dict_list(
    lst: Sequence[Mapping[K, V]],
    key_mode: Literal["intersect", "same"],
    default_val: Any = None,
) -> Dict[K, List[V]]:
    ...


@overload
def list_dict_to_dict_list(
    lst: Sequence[Mapping[K, V]],
    key_mode: Literal["union"] = "union",
    default_val: W = None,
) -> Dict[K, List[Union[V, W]]]:
    ...


def list_dict_to_dict_list(
    lst: Sequence[Mapping[K, V]],
    key_mode: KeyMode = "union",
    default_val: W = None,
) -> Dict[K, List[Union[V, W]]]:
    """Convert list of dicts to dict of lists.

    Args:
        lst: The list of dict to merge.
        key_mode: Can be "same" or "intersect".
            If "same", all the dictionaries must contains the same keys otherwise a ValueError will be raised.
            If "intersect", only the intersection of all keys will be used in output.
            If "union", the output dict will contains the union of all keys, and the missing value will use the argument default_val.
        default_val: Default value of an element when key_mode is "union". defaults to None.
    """
    if len(lst) <= 0:
        return {}

    keys = set(lst[0].keys())
    if key_mode == "same":
        if not all(keys == set(item.keys()) for item in lst[1:]):
            raise ValueError("Invalid keys for batch.")
    elif key_mode == "intersect":
        keys = intersect_lists([item.keys() for item in lst])
    elif key_mode == "union":
        keys = union_lists([item.keys() for item in lst])
    else:
        raise ValueError(
            f"Invalid argument key_mode={key_mode}. (expected one of {KEY_MODES})"
        )

    return {key: [item.get(key, default_val) for item in lst] for key in keys}


def intersect_lists(lst_of_lst: Sequence[Iterable[T]]) -> List[T]:
    """Performs intersection of elements in lists (like set intersection), but keep their original order."""
    if len(lst_of_lst) <= 0:
        return []
    out = list(dict.fromkeys(lst_of_lst[0]))
    for lst_i in lst_of_lst[1:]:
        out = [name for name in out if name in lst_i]
        if len(out) == 0:
            break
    return out


def union_lists(lst_of_lst: Iterable[Iterable[T]]) -> List[T]:
    """Performs union of elements in lists (like set union), but keep their original order."""
    out = {}
    for lst_i in lst_of_lst:
        out.update(dict.fromkeys(lst_i))
    out = list(out)
    return out
