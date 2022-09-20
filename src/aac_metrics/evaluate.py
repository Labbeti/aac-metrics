#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv

from argparse import ArgumentParser, Namespace

import yaml

from aac_metrics.functional.evaluate import evaluate


def _get_main_evaluate_args() -> Namespace:
    parser = ArgumentParser(description="Evaluate an output file.")

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="The input file containing the candidates and references.",
    )
    parser.add_argument(
        "--cand_columns",
        "-cc",
        type=str,
        nargs="+",
        default=("preds",),
        help="The column names of the candidates in the CSV file.",
    )
    parser.add_argument(
        "--mrefs_columns",
        "-mc",
        type=str,
        nargs="+",
        default=(
            "captions",
            "caption_1",
            "caption_2",
            "caption_3",
            "caption_4",
            "caption_5",
        ),
        help="The column names of the candidates in the CSV file.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="Verbose level."
    )

    args = parser.parse_args()
    return args


def _load_file(args: Namespace) -> tuple[list[str], list[list[str]]]:
    with open(args.input, "r") as file:
        reader = csv.DictReader(file)
        data = list(reader)

    candidates = []
    mult_references = []

    for data_i in data:
        cand = None
        for column in args.cand_columns:
            if column in data_i:
                cand = data_i[column]
                break

        if cand is None:
            raise ValueError("Cannot find candidate in file.")

        refs = []
        for column in args.mrefs_columns:
            if column in data_i:
                ref = data_i[column]
                ref = eval(ref)
                if isinstance(ref, str):
                    refs.append(ref)
                elif isinstance(ref, list):
                    refs += ref
                else:
                    raise ValueError(f"Invalid references type {type(ref)}.")
        candidates.append(cand)
        mult_references.append(refs)

    return candidates, mult_references


def _main_evaluate() -> None:
    args = _get_main_evaluate_args()
    print(args)
    candidates, mult_references = _load_file(args)
    global_score, local_scores = evaluate(
        candidates,
        mult_references,
        True,
        "aac",
        args.java_path,
        args.tmp_path,
        args.cache_path,
        args.verbose,
    )

    print(yaml.dump(global_score, sort_keys=False))


if __name__ == "__main__":
    _main_evaluate()
