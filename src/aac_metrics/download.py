#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import os.path as osp
import subprocess
import sys

from argparse import ArgumentParser, Namespace
from subprocess import CalledProcessError

from torch.hub import download_url_to_file


logger = logging.getLogger(__name__)


JAR_URLS = {
    "meteor": {
        "url": "https://github.com/tylin/coco-caption/raw/master/pycocoevalcap/meteor/meteor-1.5.jar",
        "fname": "meteor-1.5.jar",
    },
    "meteor_data": {
        "url": "https://github.com/tylin/coco-caption/raw/master/pycocoevalcap/meteor/data/paraphrase-en.gz",
        "fname": osp.join("data", "paraphrase-en.gz"),
    },
    "spice": {
        "url": "https://github.com/tylin/coco-caption/raw/master/pycocoevalcap/spice/spice-1.0.jar",
        "fname": "spice-1.0.jar",
    },
    "stanford_nlp": {
        "url": "https://github.com/tylin/coco-caption/raw/master/pycocoevalcap/tokenizer/stanford-corenlp-3.4.1.jar",
        "fname": "stanford-corenlp-3.4.1.jar",
    },
}


def download(
    cache_path: str = "$HOME/aac-metrics-cache",
    verbose: int = 0,
) -> None:
    """Download the code needed for SPICE, METEOR and PTB Tokenizer.

    :param cache_path: The path to the external code directory. defaults to "$HOME/aac-metrics-cache".
    :param verbose: The verbose level. defaults to 0.
    """
    cache_path = osp.expandvars(cache_path)

    stanford_nlp_dpath = osp.join(cache_path, "stanford_nlp")
    os.makedirs(stanford_nlp_dpath, exist_ok=True)

    name = "stanford_nlp"
    info = JAR_URLS[name]
    url = info["url"]
    fname = info["fname"]
    fpath = osp.join(stanford_nlp_dpath, fname)
    if not osp.isfile(fpath):
        if verbose >= 1:
            logger.info(
                f"Downloading jar source for '{name}' in directory {stanford_nlp_dpath}."
            )
        download_url_to_file(url, fpath, progress=verbose >= 1)
    else:
        if verbose >= 1:
            logger.info(f"Stanford model file '{name}' is already downloaded.")

    meteor_dpath = osp.join(cache_path, "meteor")
    os.makedirs(meteor_dpath, exist_ok=True)

    for name in ("meteor", "meteor_data"):
        info = JAR_URLS[name]
        url = info["url"]
        fname = info["fname"]
        subdir = osp.dirname(fname)
        fpath = osp.join(meteor_dpath, fname)

        if not osp.isfile(fpath):
            if verbose >= 1:
                logger.info(
                    f"Downloading jar source for '{name}' in directory {meteor_dpath}."
                )
            if subdir not in ("", "."):
                os.makedirs(osp.join(meteor_dpath, subdir), exist_ok=True)

            download_url_to_file(
                url,
                fpath,
                progress=verbose >= 1,
            )
        else:
            if verbose >= 1:
                logger.info(f"Meteor file '{name}' is already downloaded.")

    script = osp.join(osp.dirname(__file__), "..", "..", "install_spice.sh")
    if not osp.isfile(script):
        raise FileNotFoundError(f"Cannot find script '{osp.basename(script)}'.")

    spice_dpath = osp.join(cache_path, "spice")
    os.makedirs(spice_dpath, exist_ok=True)

    if verbose >= 1:
        logger.info(f"Downloading JAR sources for SPICE metric into '{spice_dpath}'...")

    command = ["bash", script, spice_dpath]
    try:
        subprocess.check_call(
            command,
            stdout=None if verbose >= 2 else subprocess.DEVNULL,
            stderr=None if verbose >= 2 else subprocess.DEVNULL,
        )
    except (CalledProcessError, PermissionError) as err:
        logger.error(err)


def _get_main_download_args() -> Namespace:
    parser = ArgumentParser(
        description="Download models and external code to evaluate captions."
    )

    parser.add_argument(
        "--cache_path",
        type=str,
        default="$HOME/aac-metrics-cache",
        help="Cache directory.",
    )
    parser.add_argument("--verbose", type=int, default=1, help="Verbose level.")

    args = parser.parse_args()
    return args


def _main_download() -> None:
    format_ = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(format_))
    pkg_logger = logging.getLogger("aac_metrics")
    pkg_logger.addHandler(handler)

    args = _get_main_download_args()

    level = logging.INFO if args.verbose <= 1 else logging.DEBUG
    pkg_logger.setLevel(level)

    download(args.cache_path, args.verbose)


if __name__ == "__main__":
    _main_download()
