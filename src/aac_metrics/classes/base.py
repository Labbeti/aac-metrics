#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Optional


try:
    from torchmetrics import Metric  # type: ignore

except ImportError:

    class Metric:
        full_state_update: Optional[bool] = False
        higher_is_better: Optional[bool] = None
        is_differentiable: Optional[bool] = False

        min_value: Optional[float] = None
        max_value: Optional[float] = None
        is_linear: Optional[bool] = None

        def compute(self) -> Any:
            return None

        def reset(self) -> None:
            pass

        def update(self, *args, **kwargs) -> None:
            pass

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            self.update(*args, **kwargs)
            outs = self.compute()
            self.reset()
            return outs
