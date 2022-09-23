#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import logging
import sys

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Iterable, Union

import yaml

from aac_metrics.functional.common import check_input
from aac_metrics.functional.evaluate import aac_evaluate


logger = logging.getLogger(__name__)


def load_csv_file(
    fpath: Union[str, Path],
    cand_columns: Iterable[str] = ("caption_predicted",),
    mrefs_columns: Iterable[str] = (
        "caption_1",
        "caption_2",
        "caption_3",
        "caption_4",
        "caption_5",
    ),
) -> tuple[list[str], list[list[str]]]:
    """Load candidates and mult_references from a CSV file."""

    with open(fpath, "r") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        data = list(reader)

    if fieldnames is None:
        raise ValueError(f"Cannot read fieldnames in CSV file {fpath=}.")

    file_cand_column = None
    for column in cand_columns:
        if column in fieldnames:
            file_cand_column = column
            break
    if file_cand_column is None:
        raise ValueError(
            f"Cannot find candidate in file. ({cand_columns=} not found in {fieldnames=})"
        )

    file_mrefs_columns = []
    for column in mrefs_columns:
        if column in fieldnames:
            file_mrefs_columns.append(column)
    if len(file_mrefs_columns) == 0:
        raise ValueError(
            f"Cannot find candidate in file. ({mrefs_columns=} not found in {fieldnames=})"
        )

    candidates = []
    mult_references = []

    for data_i in data:
        cand = data_i[file_cand_column]
        raw_refs = [data_i[column] for column in file_mrefs_columns]
        refs = []
        for ref in raw_refs:
            # Refs columns can be list[str]
            if "[" in ref and "]" in ref:
                try:
                    ref = eval(ref)
                    assert isinstance(ref, list) and all(
                        isinstance(ref_i, str) for ref_i in ref
                    )
                    refs += ref
                except (SyntaxError, NameError):
                    refs.append(ref)
            else:
                refs.append(ref)

        candidates.append(cand)
        mult_references.append(refs)

    return candidates, mult_references


def _get_main_evaluate_args() -> Namespace:
    parser = ArgumentParser(description="Evaluate an output file.")

    parser.add_argument(
        "--input_file",
        "-i",
        type=str,
        default="",
        help="The input file containing the candidates and references.",
        required=True,
    )
    parser.add_argument(
        "--cand_columns",
        "-cc",
        type=str,
        nargs="+",
        default=("caption_predicted", "preds"),
        help="The column names of the candidates in the CSV file.",
    )
    parser.add_argument(
        "--mrefs_columns",
        "-rc",
        type=str,
        nargs="+",
        default=(
            "caption_1",
            "caption_2",
            "caption_3",
            "caption_4",
            "caption_5",
            "captions",
        ),
        help="The column names of the candidates in the CSV file.",
    )
    parser.add_argument(
        "--java_path", type=str, default="java", help="Java executable path."
    )
    parser.add_argument(
        "--tmp_path", type=str, default="/tmp", help="Temporary directory path."
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default="$HOME/aac-metrics-cache",
        help="Cache directory path.",
    )
    parser.add_argument("--verbose", type=int, default=0, help="Verbose level.")

    args = parser.parse_args()
    return args


def _main_evaluate() -> None:
    format_ = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(format_))
    pkg_logger = logging.getLogger("aac_metrics")
    pkg_logger.addHandler(handler)

    args = _get_main_evaluate_args()

    level = logging.INFO if args.verbose <= 1 else logging.DEBUG
    pkg_logger.setLevel(level)

    logger.info(f"Load file {args.input_file}...")

    candidates, mult_references = load_csv_file(
        args.input_file, args.cand_columns, args.mrefs_columns
    )
    check_input(candidates, mult_references)

    refs_lens = list(map(len, mult_references))
    logger.info(
        f"Found {len(candidates)} candidates and references and [{min(refs_lens)}, {max(refs_lens)}] references per candidate."
    )

    global_score, _local_scores = aac_evaluate(
        candidates,
        mult_references,
        True,
        args.java_path,
        args.tmp_path,
        args.cache_path,
        args.verbose,
    )
    from torch import Tensor

    global_score = {
        k: v.item() if isinstance(v, Tensor) else v for k, v in global_score.items()
    }
    logger.info(f"Global scores:\n{yaml.dump(global_score, sort_keys=False)}")


if __name__ == "__main__":
    _main_evaluate()
