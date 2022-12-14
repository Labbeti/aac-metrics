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

from aac_metrics.classes.fense import FENSE
from aac_metrics.functional.meteor import FNAME_METEOR_JAR
from aac_metrics.functional.spice import FNAME_SPICE_JAR, DNAME_SPICE_CACHE
from aac_metrics.utils.tokenization import FNAME_STANFORD_CORENLP_3_4_1_JAR


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
    cache_path: str = "$HOME/.cache",
    tmp_path: str = "/tmp",
    verbose: int = 0,
) -> None:
    """Download the code needed for SPICE, METEOR and PTB Tokenizer.

    :param cache_path: The path to the external code directory. defaults to "$HOME/.cache".
    :param verbose: The verbose level. defaults to 0.
    """
    cache_path = osp.expandvars(cache_path)
    tmp_path = osp.expandvars(tmp_path)

    os.makedirs(cache_path, exist_ok=True)
    os.makedirs(tmp_path, exist_ok=True)

    # Download JAR file for tokenization
    stanford_nlp_dpath = osp.join(
        cache_path, osp.dirname(FNAME_STANFORD_CORENLP_3_4_1_JAR)
    )
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

    # Download JAR files for METEOR metric
    meteor_dpath = osp.join(cache_path, osp.dirname(FNAME_METEOR_JAR))
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

    # Download JAR files for SPICE metric
    spice_jar_dpath = osp.join(cache_path, osp.dirname(FNAME_SPICE_JAR))
    os.makedirs(spice_jar_dpath, exist_ok=True)

    spice_cache_path = osp.join(cache_path, DNAME_SPICE_CACHE)
    os.makedirs(spice_cache_path, exist_ok=True)

    script_path = osp.join(osp.dirname(__file__), "..", "..", "install_spice.sh")
    if not osp.isfile(script_path):
        raise FileNotFoundError(f"Cannot find script '{osp.basename(script_path)}'.")

    if verbose >= 1:
        logger.info(
            f"Downloading JAR sources for SPICE metric into '{spice_jar_dpath}'..."
        )

    command = ["bash", script_path, spice_jar_dpath]
    try:
        subprocess.check_call(
            command,
            stdout=None if verbose >= 2 else subprocess.DEVNULL,
            stderr=None if verbose >= 2 else subprocess.DEVNULL,
        )
    except (CalledProcessError, PermissionError) as err:
        logger.error(err)

    # Download models files for FENSE metric
    if verbose >= 1:
        logger.info("Downloading sBert and Bert error detector for FENSE metric...")
    _ = FENSE(device="cpu")


def _get_main_download_args() -> Namespace:
    parser = ArgumentParser(
        description="Download models and external code to evaluate captions."
    )

    parser.add_argument(
        "--cache_path",
        type=str,
        default="$HOME/.cache",
        help="Cache directory path.",
    )
    parser.add_argument(
        "--tmp_path",
        type=str,
        default="/tmp",
        help="Temporary directory path.",
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

    download(args.cache_path, args.tmp_path, args.verbose)


if __name__ == "__main__":
    _main_download()
