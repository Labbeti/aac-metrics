#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Automated Audio Captioning metrics.
"""

__name__ = "aac-metrics"
__author__ = "Etienne Labbé (Labbeti)"
__license__ = "MIT"
__maintainer__ = "Etienne Labbé (Labbeti)"
__status__ = "Development"
__version__ = "0.1.0"


from .evaluate import evaluate
from .classes import (
    CocoBLEU,
    CocoCIDErD,
    CocoMETEOR,
    CocoRougeL,
    CocoSPICE,
    SPIDEr,
    DiversityRatio,
)
