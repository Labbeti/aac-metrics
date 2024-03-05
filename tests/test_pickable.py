#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import unittest
from unittest import TestCase

from aac_metrics.classes.evaluate import Evaluate


class TestPickle(TestCase):
    # Tests methods
    def test_pickle_dump(self) -> None:
        metrics = Evaluate(metrics="all")
        try:
            pickle.dumps(metrics)
        except pickle.PicklingError:
            self.assertTrue(False, f"Cannot pickle {metrics.__class__.__name__}.")


if __name__ == "__main__":
    unittest.main()
