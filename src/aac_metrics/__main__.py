#!/usr/bin/env python
# -*- coding: utf-8 -*-


def _print_usage() -> None:
    print(
        "Command line usage :\n"
        "- Download models and external code               : aac-met-download ...\n"
        "- Print scores from candidate and references file : aac-met-evaluate -i [FILEPATH]\n"
        "- Print package version                           : aac-met-info\n"
        "- Show this usage page                            : aac-met\n"
    )


if __name__ == "__main__":
    _print_usage()
