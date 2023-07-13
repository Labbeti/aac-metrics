#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Generic, Optional, TypeVar

from torch import nn

OutType = TypeVar("OutType")


class AACMetric(nn.Module, Generic[OutType]):
    """Base Metric module for AAC metrics. Similar to torchmetrics.Metric."""

    # Global values
    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = None
    is_differentiable: Optional[bool] = False

    # The theorical minimal value of the main global score of the metric.
    min_value: Optional[float] = None
    # The theorical maximal value of the main global score of the metric.
    max_value: Optional[float] = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    # Public methods
    def compute(self) -> OutType:
        return None  # type: ignore

    def forward(self, *args: Any, **kwargs: Any) -> OutType:
        self.update(*args, **kwargs)
        output = self.compute()
        self.reset()
        return output

    def reset(self) -> None:
        pass

    def update(self, *args, **kwargs) -> None:
        pass

    # Magic methods
    def __call__(self, *args: Any, **kwds: Any) -> OutType:
        return super().__call__(*args, **kwds)
