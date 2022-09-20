#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any


try:
    from torchmetrics import Metric  # type: ignore

except ImportError:

    class Metric:
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
