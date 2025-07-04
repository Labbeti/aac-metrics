#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import unittest
from unittest import TestCase

import pythonwrench as pw

from aac_metrics.utils.checks import (
    MAX_JAVA_MAJOR_VERSION,
    MIN_JAVA_MAJOR_VERSION,
    _check_java_version,
    is_mono_sents,
    is_mult_sents,
)


class TestUtils(TestCase):
    # Tests methods
    def test_misc_functions_1(self) -> None:
        lst = [
            list(map(str, range(random.randint(0, 100))))
            for _ in range(random.randint(0, 10))
        ]
        for sublst in lst:
            random.shuffle(sublst)
        random.shuffle(lst)

        self.assertTrue(is_mult_sents(lst))

        flatten, sizes = pw.flat_list_of_list(lst)

        self.assertTrue(is_mono_sents(flatten))
        self.assertEqual(len(lst), len(sizes))
        self.assertEqual(len(flatten), sum(sizes))

        unflat = pw.unflat_list_of_list(flatten, sizes)

        self.assertTrue(is_mult_sents(unflat))
        self.assertEqual(len(lst), len(unflat))
        self.assertListEqual(lst, unflat)

    def test_check_java_versions(self) -> None:
        test_set = [
            ("1.0.0", False),
            ("1.7.0", False),
            ("1.8.0", True),
            ("1.9.0", False),
            ("1.10.0", False),
            ("9.0.0", True),
            ("10.0.0", True),
            ("11.0.0", True),
            ("12.0.0", True),
            ("13.0.0", True),
            ("14.0.0", False),
            ("17.0.0", False),
            ("20.0.0", False),
        ]
        for version, expected in test_set:
            output = _check_java_version(
                version, MIN_JAVA_MAJOR_VERSION, MAX_JAVA_MAJOR_VERSION
            )
            self.assertEqual(output, expected, f"{version=}")


if __name__ == "__main__":
    unittest.main()
