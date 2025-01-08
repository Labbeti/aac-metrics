#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import unittest
from pickle import PicklingError
from unittest import TestCase

from aac_metrics.classes import MACE, CLAPSim, Evaluate


class TestPickle(TestCase):
    # Tests methods
    def test_pickle_dump(self) -> None:
        metrics = Evaluate(metrics="all")

        # Exclude non-pickable metrics
        for i, metric in enumerate(metrics):
            if isinstance(metric, (CLAPSim, MACE)):
                metrics.pop(i)

        try:
            pickle.dumps(metrics)
        except PicklingError as err:
            msg = f"Cannot pickle {metrics.__class__.__name__}.\n{err}"
            self.assertTrue(False, msg)


if __name__ == "__main__":
    unittest.main()
