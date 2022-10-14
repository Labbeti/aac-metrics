#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import logging
import sys

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Iterable, Union

import yaml

from aac_metrics.functional.common import _check_input
from aac_metrics.functional.evaluate import aac_evaluate
from aac_metrics.utils.misc import _check_java_path


logger = logging.getLogger(__name__)


def load_csv_file(
    fpath: Union[str, Path],
    cands_columns: Iterable[str] = ("caption_predicted",),
    mrefs_columns: Iterable[str] = (
        "caption_1",
        "caption_2",
        "caption_3",
        "caption_4",
        "caption_5",
    ),
    load_mult_cands: bool = False,
) -> tuple[list, list[list[str]]]:
    """Load candidates and mult_references from a CSV file.

    :param fpath: The filepath to the CSV file.
    :param cands_columns: The columns of the candidates. defaults to ("captions_predicted",).
    :param mrefs_columns: The columns of the multiple references. defaults to ("caption_1", "caption_2", "caption_3", "caption_4", "caption_5").
    :param load_mult_cands: If True, load multiple candidates from file. defaults to False.
    :returns: A tuple of (candidates, mult_references) loaded from file.
    """

    with open(fpath, "r") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        data = list(reader)

    if fieldnames is None:
        raise ValueError(f"Cannot read fieldnames in CSV file {fpath=}.")

    file_cands_columns = [column for column in cands_columns if column in fieldnames]
    file_mrefs_columns = [column for column in mrefs_columns if column in fieldnames]

    if (load_mult_cands and len(file_cands_columns) <= 0) or (
        not load_mult_cands and len(file_cands_columns) != 1
    ):
        raise ValueError(
            f"Cannot find candidate in file. ({cands_columns=} not found in {fieldnames=})"
        )
    if len(file_mrefs_columns) <= 0:
        raise ValueError(
            f"Cannot find candidate in file. ({mrefs_columns=} not found in {fieldnames=})"
        )

    if load_mult_cands:
        mult_candidates = _load_columns(data, file_cands_columns)
        mult_references = _load_columns(data, file_mrefs_columns)
        return mult_candidates, mult_references
    else:
        file_cand_column = file_cands_columns[0]
        candidates = [data_i[file_cand_column] for data_i in data]
        mult_references = _load_columns(data, file_mrefs_columns)
        return candidates, mult_references


def _load_columns(data: list[dict[str, str]], columns: list[str]) -> list[list[str]]:
    mult_sentences = []
    for data_i in data:
        raw_sents = [data_i[column] for column in columns]
        sents = []
        for sent in raw_sents:
            # Refs columns can be list[str]
            if "[" in sent and "]" in sent:
                try:
                    sent = eval(sent)
                    assert isinstance(sent, list) and all(
                        isinstance(ref_i, str) for ref_i in sent
                    )
                    sents += sent
                except (SyntaxError, NameError):
                    sents.append(sent)
            else:
                sents.append(sent)

        mult_sentences.append(sents)
    return mult_sentences


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
        "--cache_path",
        type=str,
        default="$HOME/aac-metrics-cache",
        help="Cache directory path.",
    )
    parser.add_argument(
        "--java_path",
        type=str,
        default="java",
        help="Java executable path.",
    )
    parser.add_argument(
        "--tmp_path",
        type=str,
        default="/tmp",
        help="Temporary directory path.",
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

    if not _check_java_path(args.java_path):
        raise RuntimeError(f"Invalid argument java_path={args.java_path}.")

    level = logging.INFO if args.verbose <= 1 else logging.DEBUG
    pkg_logger.setLevel(level)

    if args.verbose >= 1:
        logger.info(f"Load file {args.input_file}...")

    candidates, mult_references = load_csv_file(
        args.input_file, args.cand_columns, args.mrefs_columns
    )
    _check_input(candidates, mult_references)

    refs_lens = list(map(len, mult_references))
    if args.verbose >= 1:
        logger.info(
            f"Found {len(candidates)} candidates, {len(mult_references)} references and [{min(refs_lens)}, {max(refs_lens)}] references per candidate."
        )

    global_scores, _local_scores = aac_evaluate(
        candidates,
        mult_references,
        True,
        args.cache_path,
        args.java_path,
        args.tmp_path,
        args.verbose,
    )

    global_scores = {k: v.item() for k, v in global_scores.items()}
    logger.info(f"Global scores:\n{yaml.dump(global_scores, sort_keys=False)}")


if __name__ == "__main__":
    _main_evaluate()
