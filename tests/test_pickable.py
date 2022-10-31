#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import unittest

from unittest import TestCase

from aac_metrics.classes.evaluate import _get_metrics_classes_list


class TestCompare(TestCase):
    def test_pickle_dump(self) -> None:
        metrics = _get_metrics_classes_list("all")

        for metric in metrics:
            try:
                pickle.dumps(metric)
            except pickle.PicklingError:
                self.assert_(False, f"Cannot pickle {metric.__class__.__name__}.")


if __name__ == "__main__":
    unittest.main()
