#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import os.path as osp
import platform
import shutil
import subprocess
import sys

from argparse import ArgumentParser, Namespace
from subprocess import CalledProcessError
from typing import Optional
from zipfile import ZipFile

from torch.hub import download_url_to_file

from aac_metrics.classes.fense import FENSE
from aac_metrics.functional.meteor import DNAME_METEOR_CACHE
from aac_metrics.functional.spice import (
    FNAME_SPICE_JAR,
    DNAME_SPICE_LOCAL_CACHE,
    DNAME_SPICE_CACHE,
)
from aac_metrics.utils.paths import (
    _get_cache_path,
    _get_tmp_path,
    get_default_cache_path,
    get_default_tmp_path,
)
from aac_metrics.utils.tokenization import FNAME_STANFORD_CORENLP_3_4_1_JAR


pylog = logging.getLogger(__name__)


DATA_URLS = {
    "meteor": {
        "url": "https://github.com/tylin/coco-caption/raw/master/pycocoevalcap/meteor/meteor-1.5.jar",
        "fname": "meteor-1.5.jar",
    },
    "meteor_data": {
        "url": "https://github.com/tylin/coco-caption/raw/master/pycocoevalcap/meteor/data/paraphrase-en.gz",
        "fname": osp.join("data", "paraphrase-en.gz"),
    },
    "meteor_data_fr": {
        "url": "https://github.com/cmu-mtlab/meteor/raw/master/data/paraphrase-fr.gz",
        "fname": osp.join("data", "paraphrase-fr.gz"),
    },
    "meteor_data_de": {
        "url": "https://github.com/cmu-mtlab/meteor/raw/master/data/paraphrase-de.gz",
        "fname": osp.join("data", "paraphrase-de.gz"),
    },
    "meteor_data_es": {
        "url": "https://github.com/cmu-mtlab/meteor/raw/master/data/paraphrase-es.gz",
        "fname": osp.join("data", "paraphrase-es.gz"),
    },
    "meteor_data_cz": {
        "url": "https://github.com/cmu-mtlab/meteor/raw/master/data/paraphrase-cz.gz",
        "fname": osp.join("data", "paraphrase-cz.gz"),
    },
    "spice": {
        "url": "https://github.com/tylin/coco-caption/raw/master/pycocoevalcap/spice/spice-1.0.jar",
        "fname": "spice-1.0.jar",
    },
    "stanford_nlp": {
        "url": "https://github.com/tylin/coco-caption/raw/master/pycocoevalcap/tokenizer/stanford-corenlp-3.4.1.jar",
        "fname": "stanford-corenlp-3.4.1.jar",
    },
    "spice_zip": {
        "url": "https://panderson.me/images/SPICE-1.0.zip",
        "fname": "SPICE-1.0.zip",
        "extract_to": ".",
    },
    "spice_corenlp_zip": {
        "url": "http://nlp.stanford.edu/software/stanford-corenlp-full-2015-12-09.zip",
        "fname": osp.join("SPICE-1.0", "stanford-corenlp-full-2015-12-09.zip"),
        "extract_to": "lib",
    },
}
_TRUE_VALUES = ("true", "1", "t")
_FALSE_VALUES = ("false", "0", "f")


def download(
    cache_path: str = ...,
    tmp_path: str = ...,
    use_shell: Optional[bool] = None,
    ptb_tokenizer: bool = True,
    meteor: bool = True,
    spice: bool = True,
    fense: bool = True,
    verbose: int = 0,
) -> None:
    """Download the code needed for SPICE, METEOR, PTB Tokenizer and FENSE.

    :param cache_path: The path to the external code directory. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_cache_path`.
    :param tmp_path: The path to a temporary directory. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_tmp_path`.
    :param use_shell: Optional argument to force use os-specific shell for the bash installation script program.
        If None, it will use shell only on Windows OS.
        defaults to None.
    :param ptb_tokenizer: If True, downloads the PTBTokenizer code in cache directory. defaults to True.
    :param meteor: If True, downloads the METEOR code in cache directory. defaults to True.
    :param spice: If True, downloads the SPICE code in cache directory. defaults to True.
    :param fense: If True, downloads the FENSE models. defaults to True.
    :param verbose: The verbose level. defaults to 0.
    """
    if verbose >= 1:
        pylog.info(f"aac-metrics download started.")

    cache_path = _get_cache_path(cache_path)
    tmp_path = _get_tmp_path(tmp_path)

    os.makedirs(cache_path, exist_ok=True)
    os.makedirs(tmp_path, exist_ok=True)

    if verbose >= 2:
        pylog.debug("AAC setup:")
        pylog.debug(f"  Cache directory: {cache_path}")
        pylog.debug(f"  Temp directory: {tmp_path}")

    if ptb_tokenizer:
        _download_ptb_tokenizer(cache_path, verbose)

    if meteor:
        _download_meteor(cache_path, verbose)

    if spice:
        _download_spice(cache_path, use_shell, verbose)

    if fense:
        _download_fense(verbose)

    if verbose >= 1:
        pylog.info(f"aac-metrics download finished.")


def _download_ptb_tokenizer(
    cache_path: str,
    verbose: int = 0,
) -> None:
    # Download JAR file for tokenization
    stanford_nlp_dpath = osp.join(
        cache_path, osp.dirname(FNAME_STANFORD_CORENLP_3_4_1_JAR)
    )
    os.makedirs(stanford_nlp_dpath, exist_ok=True)

    name = "stanford_nlp"
    info = DATA_URLS[name]
    url = info["url"]
    fname = info["fname"]
    fpath = osp.join(stanford_nlp_dpath, fname)

    if not osp.isfile(fpath):
        if verbose >= 1:
            pylog.info(
                f"Downloading JAR source for '{name}' in directory {stanford_nlp_dpath}."
            )
        download_url_to_file(url, fpath, progress=verbose >= 1)
    else:
        if verbose >= 1:
            pylog.info(f"Stanford model file '{name}' is already downloaded.")


def _download_meteor(
    cache_path: str,
    verbose: int = 0,
) -> None:
    # Download JAR files for METEOR metric
    meteor_dpath = osp.join(cache_path, DNAME_METEOR_CACHE)
    os.makedirs(meteor_dpath, exist_ok=True)

    meteors_names = [name for name in DATA_URLS.keys() if name.startswith("meteor")]

    for name in meteors_names:
        info = DATA_URLS[name]
        url = info["url"]
        fname = info["fname"]
        subdir = osp.dirname(fname)
        fpath = osp.join(meteor_dpath, fname)

        if osp.isfile(fpath):
            if verbose >= 1:
                pylog.info(f"Meteor file '{name}' is already downloaded.")
            continue

        if verbose >= 1:
            pylog.info(f"Downloading source for '{fname}' in directory {meteor_dpath}.")
        if subdir not in ("", "."):
            os.makedirs(osp.dirname(fpath), exist_ok=True)

        download_url_to_file(
            url,
            fpath,
            progress=verbose >= 1,
        )


def _download_spice(
    cache_path: str,
    use_shell: Optional[bool] = None,
    verbose: int = 0,
) -> None:
    # Download JAR files for SPICE metric
    spice_cache_dpath = osp.join(cache_path, DNAME_SPICE_CACHE)
    spice_jar_dpath = osp.join(cache_path, osp.dirname(FNAME_SPICE_JAR))
    spice_local_cache_path = osp.join(cache_path, DNAME_SPICE_LOCAL_CACHE)

    os.makedirs(spice_jar_dpath, exist_ok=True)
    os.makedirs(spice_local_cache_path, exist_ok=True)

    for name in ("spice_zip", "spice_corenlp_zip"):
        url = DATA_URLS[name]["url"]
        fname = DATA_URLS[name]["fname"]
        fpath = osp.join(spice_cache_dpath, fname)

        if osp.isfile(fpath):
            if verbose >= 1:
                pylog.info(f"File '{fpath}' is already downloaded for SPICE.")
        else:
            if verbose >= 1:
                pylog.info(f"Downloading file '{fpath}' for SPICE...")

            dpath = osp.dirname(fpath)
            os.makedirs(dpath, exist_ok=True)
            download_url_to_file(url, fpath, progress=verbose > 0)

        if fname.endswith(".zip"):
            target_dpath = osp.join(spice_cache_dpath, DATA_URLS[name]["extract_to"])
            os.makedirs(target_dpath, exist_ok=True)
            with ZipFile(fpath, "r") as file:
                file.extractall(target_dpath)

    # TODO: rm
    # mv $SPICELIB/$CORENLP/stanford-corenlp-3.6.0.jar $SPICELIB/
    # mv $SPICELIB/$CORENLP/stanford-corenlp-3.6.0-models.jar $SPICELIB/
    # dpath_unzip="$dpath_spice/SPICE-1.0"
    # mv "$dpath_unzip/lib" "$dpath_spice"
    # mv "$dpath_unzip/spice-1.0.jar" "$dpath_spice"

    spice_lib_dpath = osp.join(spice_cache_dpath, "lib")
    spice_unzip_dpath = osp.join(spice_cache_dpath, "SPICE-1.0")
    corenlp_dname = "stanford-corenlp-full-2015-12-09"

    to_move = {
        osp.join(
            spice_lib_dpath, corenlp_dname, "stanford-corenlp-3.6.0.jar"
        ): spice_lib_dpath,
        osp.join(
            spice_lib_dpath, corenlp_dname, "stanford-corenlp-3.6.0-models.jar"
        ): spice_lib_dpath,
        osp.join(spice_unzip_dpath, "lib"): spice_cache_dpath,
        osp.join(spice_unzip_dpath, "spice-1.0.jar"): spice_cache_dpath,
    }
    for source, target in to_move.items():
        if not osp.exists(target):
            shutil.move(source, target)

    # TODO: rm
    # script_fname = "install_spice.sh"
    # script_fpath = osp.join(osp.dirname(__file__), script_fname)
    # if not osp.isfile(script_fpath):
    #     raise FileNotFoundError(f"Cannot find script '{osp.basename(script_fpath)}'.")

    # if verbose >= 1:
    #     pylog.info(
    #         f"Downloading JAR sources for SPICE metric into '{spice_jar_dpath}'..."
    #     )

    # if use_shell is None:
    #     use_shell = platform.system() == "Windows"

    # command = ["bash", script_fpath, spice_jar_dpath]
    # try:
    #     subprocess.check_call(
    #         command,
    #         stdout=None if verbose >= 1 else subprocess.DEVNULL,
    #         stderr=None if verbose >= 1 else subprocess.DEVNULL,
    #         shell=use_shell,
    #     )
    # except (CalledProcessError, PermissionError) as err:
    #     pylog.error(
    #         f"Cannot install SPICE java source code from '{script_fname}' script."
    #     )
    #     raise err


def _download_fense(
    verbose: int = 0,
) -> None:
    # Download models files for FENSE metric
    if verbose >= 1:
        pylog.info("Downloading SBERT and BERT error detector for FENSE metric...")
    _ = FENSE(device="cpu")


def _get_main_download_args() -> Namespace:
    parser = ArgumentParser(
        description="Download models and external code to evaluate captions."
    )

    parser.add_argument(
        "--cache_path",
        type=str,
        default=get_default_cache_path(),
        help="Cache directory path.",
    )
    parser.add_argument(
        "--tmp_path",
        type=str,
        default=get_default_tmp_path(),
        help="Temporary directory path.",
    )
    parser.add_argument(
        "--ptb_tokenizer",
        type=_str_to_bool,
        default=True,
        help="Download PTBTokenizer Java source code.",
    )
    parser.add_argument(
        "--meteor",
        type=_str_to_bool,
        default=True,
        help="Download METEOR Java source code.",
    )
    parser.add_argument(
        "--spice",
        type=_str_to_bool,
        default=True,
        help="Download SPICE Java source code.",
    )
    parser.add_argument(
        "--fense",
        type=_str_to_bool,
        default=True,
        help="Download FENSE models.",
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

    download(
        cache_path=args.cache_path,
        tmp_path=args.tmp_path,
        ptb_tokenizer=args.ptb_tokenizer,
        meteor=args.meteor,
        spice=args.spice,
        fense=args.fense,
        verbose=args.verbose,
    )


def _str_to_bool(s: str) -> bool:
    s = str(s).strip().lower()
    if s in _TRUE_VALUES:
        return True
    elif s in _FALSE_VALUES:
        return False
    else:
        raise ValueError(
            f"Invalid argument {s=}. (expected one of {_TRUE_VALUES + _FALSE_VALUES})"
        )


if __name__ == "__main__":
    _main_download()
