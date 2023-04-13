#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import unittest

from unittest import TestCase

from aac_metrics.utils.checks import is_mono_sents, is_mult_sents
from aac_metrics.utils.collections import flat_list, unflat_list


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

        flatten, sizes = flat_list(lst)

        self.assertTrue(is_mono_sents(flatten))
        self.assertEqual(len(lst), len(sizes))
        self.assertEqual(len(flatten), sum(sizes))

        unflat = unflat_list(flatten, sizes)

        self.assertTrue(is_mult_sents(unflat))
        self.assertEqual(len(lst), len(unflat))
        self.assertListEqual(lst, unflat)


if __name__ == "__main__":
    unittest.main()
