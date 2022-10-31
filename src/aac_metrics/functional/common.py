#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any


def is_mono_sents(sents: Any) -> bool:
    """Returns True if input is list[str]."""
    return isinstance(sents, list) and all(isinstance(sent, str) for sent in sents)


def is_mult_sents(mult_sents: Any) -> bool:
    """Returns True if input is list[list[str]]."""
    return (
        isinstance(mult_sents, list)
        and all(isinstance(sents, list) for sents in mult_sents)
        and all(isinstance(sent, str) for sents in mult_sents for sent in sents)
    )


def _check_input(
    candidates: Any,
    mult_references: Any,
) -> None:
    """Returns True candidates and mult_references have a valid type and size."""
    if not is_mono_sents(candidates):
        raise ValueError("Invalid candidates type. (expected list[str])")

    if not is_mult_sents(mult_references):
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
