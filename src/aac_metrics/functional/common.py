#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any


def _check_input(
    candidates: Any,
    mult_references: Any,
) -> None:
    """Returns True candidates and mult_references have a valid type and size."""
    cands_is_list = isinstance(candidates, list)
    cand_is_str = all(isinstance(cand, str) for cand in candidates)
    if not all((cands_is_list, cand_is_str)):
        raise ValueError("Invalid candidates type. (expected list[str])")

    mrefs_is_list = isinstance(mult_references, list)
    refs_is_list = all(isinstance(refs, list) for refs in mult_references)
    ref_is_str = all(isinstance(ref, str) for refs in mult_references for ref in refs)
    if not all((mrefs_is_list, refs_is_list, ref_is_str)):
        raise ValueError("Invalid mult_references type. (expected list[list[str]])")

    same_len = len(candidates) == len(mult_references)
    if not same_len:
        raise ValueError(
            f"Invalid number of candidates ({len(candidates)}) with the number of references ({len(mult_references)})."
        )

    at_least_1_ref_per_cand = all(len(refs) > 0 for refs in mult_references)
    if not at_least_1_ref_per_cand:
        raise ValueError(
            "Invalid number of references per candidate. (found at least 1 empty list of references)"
        )
