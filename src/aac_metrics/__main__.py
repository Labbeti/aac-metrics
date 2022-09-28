#!/usr/bin/env python
# -*- coding: utf-8 -*-


def _print_usage() -> None:
    print(
        "Command line usage :\n"
        "- Download models and external code               : aac-metrics-download ...\n"
        "- Print scores from candidate and references file : aac-metrics-evaluate -i [FILEPATH]\n"
        "- Print package version                           : aac-metrics-info\n"
        "- Show this usage page                            : aac-metrics\n"
    )


if __name__ == "__main__":
    _print_usage()
