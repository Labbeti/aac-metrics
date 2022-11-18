#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Optional

from torch import nn


try:
    from torchmetrics import Metric  # type: ignore

except ModuleNotFoundError:

    class Metric(nn.Module):
        """Base Metric module used when torchmetrics is not installed."""

        # Global values
        full_state_update: Optional[bool] = False
        higher_is_better: Optional[bool] = None
        is_differentiable: Optional[bool] = False

        # The theorical minimal value of the main global score of the metric.
        min_value: Optional[float] = None
        # The theorical maximal value of the main global score of the metric.
        max_value: Optional[float] = None

        # Public methods
        def compute(self) -> Any:
            return None

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            self.update(*args, **kwargs)
            output = self.compute()
            self.reset()
            return output

        def reset(self) -> None:
            pass

        def update(self, *args, **kwargs) -> None:
            pass
