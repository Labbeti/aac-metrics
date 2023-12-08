#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import unittest

from unittest import TestCase

from aac_metrics.classes.evaluate import _instantiate_metrics_classes


class TestCompare(TestCase):
    # Tests methods
    def test_pickle_dump(self) -> None:
        metrics = _instantiate_metrics_classes("all")

        for metric in metrics:
            try:
                pickle.dumps(metric)
            except pickle.PicklingError:
                self.assertTrue(False, f"Cannot pickle {metric.__class__.__name__}.")


if __name__ == "__main__":
    unittest.main()
