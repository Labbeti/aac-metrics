#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Optional

from torch import Tensor

from aac_metrics.functional.coco_spider import coco_spider
from aac_metrics.functional.mult_cands import mult_cands_wrapper


def coco_spider_max(
    mult_candidates: list[list[str]],
    mult_references: list[list[str]],
    cider_kwargs: Optional[dict[str, Any]] = None,
    spice_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    return mult_cands_wrapper(
        coco_spider,
        mult_candidates,
        mult_references,
        reduction="max",
        cider_kwargs=cider_kwargs,
        spice_kwargs=spice_kwargs,
    )
