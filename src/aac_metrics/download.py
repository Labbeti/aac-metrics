#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import os.path as osp
import shutil

from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Union
from zipfile import ZipFile

from torch.hub import download_url_to_file

import aac_metrics

from aac_metrics.classes.bert_score_mrefs import BERTScoreMRefs
from aac_metrics.classes.fense import FENSE
from aac_metrics.functional.meteor import DNAME_METEOR_CACHE
from aac_metrics.functional.spice import (
    DNAME_SPICE_CACHE,
    DNAME_SPICE_LOCAL_CACHE,
    FNAME_SPICE_JAR,
    check_spice_install,
)
from aac_metrics.utils.cmdline import _str_to_bool, _setup_logging
from aac_metrics.utils.globals import (
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
    },
    "spice_corenlp_zip": {
        "url": "http://nlp.stanford.edu/software/stanford-corenlp-full-2015-12-09.zip",
        "fname": osp.join("SPICE-1.0", "stanford-corenlp-full-2015-12-09.zip"),
    },
}


def download_metrics(
    cache_path: Union[str, Path, None] = None,
    tmp_path: Union[str, Path, None] = None,
    clean_archives: bool = True,
    ptb_tokenizer: bool = True,
    meteor: bool = True,
    spice: bool = True,
    fense: bool = True,
    bert_score: bool = True,
    verbose: int = 0,
) -> None:
    """Download the code needed for SPICE, METEOR, PTB Tokenizer and FENSE.

    :param cache_path: The path to the external code directory. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_cache_path`.
    :param tmp_path: The path to a temporary directory. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_tmp_path`.
    :param clean_archives: If True, remove all archives files. defaults to True.
    :param ptb_tokenizer: If True, downloads the PTBTokenizer code in cache directory. defaults to True.
    :param meteor: If True, downloads the METEOR code in cache directory. defaults to True.
    :param spice: If True, downloads the SPICE code in cache directory. defaults to True.
    :param fense: If True, downloads the FENSE models. defaults to True.
    :param bert_score: If True, downloads the BERTScore model. defaults to True.
    :param verbose: The verbose level. defaults to 0.
    """
    if verbose >= 1:
        pylog.info("aac-metrics download started.")

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
        _download_spice(cache_path, clean_archives, verbose)

    if fense:
        _download_fense(verbose)

    if bert_score:
        _download_bert_score(verbose)

    if verbose >= 1:
        pylog.info("aac-metrics download finished.")


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
    clean_archives: bool = True,
    verbose: int = 0,
) -> None:
    """Download SPICE java code.

    Target SPICE directory tree:

    spice
    ├── cache
    ├── lib
    │   ├── ejml-0.23.jar
    │   ├── fst-2.47.jar
    │   ├── guava-19.0.jar
    │   ├── hamcrest-core-1.3.jar
    │   ├── jackson-core-2.5.3.jar
    │   ├── javassist-3.19.0-GA.jar
    │   ├── json-simple-1.1.1.jar
    │   ├── junit-4.12.jar
    │   ├── lmdbjni-0.4.6.jar
    │   ├── lmdbjni-linux64-0.4.6.jar
    │   ├── lmdbjni-osx64-0.4.6.jar
    │   ├── lmdbjni-win64-0.4.6.jar
    │   ├── Meteor-1.5.jar
    │   ├── objenesis-2.4.jar
    │   ├── SceneGraphParser-1.0.jar
    │   ├── slf4j-api-1.7.12.jar
    │   ├── slf4j-simple-1.7.21.jar
    │   ├── stanford-corenlp-3.6.0.jar
    │   └── stanford-corenlp-3.6.0-models.jar
    └── spice-1.0.jar
    """
    try:
        check_spice_install(cache_path)
        return None
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        pass

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
            if verbose >= 1:
                pylog.info(f"Extracting {fname} to {spice_cache_dpath}...")

            with ZipFile(fpath, "r") as file:
                file.extractall(spice_cache_dpath)

    spice_lib_dpath = osp.join(spice_cache_dpath, "lib")
    spice_unzip_dpath = osp.join(spice_cache_dpath, "SPICE-1.0")
    corenlp_dpath = osp.join(spice_cache_dpath, "stanford-corenlp-full-2015-12-09")

    # Note: order matter here
    to_move = [
        ("f", osp.join(spice_unzip_dpath, "spice-1.0.jar"), spice_cache_dpath),
        ("f", osp.join(corenlp_dpath, "stanford-corenlp-3.6.0.jar"), spice_lib_dpath),
        (
            "f",
            osp.join(corenlp_dpath, "stanford-corenlp-3.6.0-models.jar"),
            spice_lib_dpath,
        ),
    ]
    for name in os.listdir(osp.join(spice_unzip_dpath, "lib")):
        if not name.endswith(".jar"):
            continue
        fpath = osp.join(spice_unzip_dpath, "lib", name)
        to_move.append(("f", fpath, spice_lib_dpath))

    os.makedirs(spice_lib_dpath, exist_ok=True)

    for i, (_src_type, src_path, parent_tgt_dpath) in enumerate(to_move):
        tgt_path = osp.join(parent_tgt_dpath, osp.basename(src_path))

        if osp.exists(tgt_path):
            if verbose >= 1:
                pylog.info(
                    f"Target '{tgt_path}' already exists. ({i+1}/{len(to_move)})"
                )
        else:
            if verbose >= 1:
                pylog.info(
                    f"Moving '{src_path}' to '{parent_tgt_dpath}'... ({i+1}/{len(to_move)})"
                )
            shutil.move(src_path, parent_tgt_dpath)

    shutil.rmtree(corenlp_dpath)
    if clean_archives:
        spice_zip_fname = DATA_URLS["spice_zip"]["fname"]
        spice_zip_fpath = osp.join(spice_cache_dpath, spice_zip_fname)

        os.remove(spice_zip_fpath)
        shutil.rmtree(spice_unzip_dpath)


def _download_fense(
    verbose: int = 0,
) -> None:
    # Download models files for FENSE metric
    if verbose >= 1:
        pylog.info("Downloading SBERT and BERT error detector for FENSE metric...")
    _ = FENSE(device="cpu")


def _download_bert_score(
    verbose: int = 0,
) -> None:
    # Download models files for BERTScore metric
    if verbose >= 1:
        pylog.info("Downloading BERT model for BERTScore metric...")
    _ = BERTScoreMRefs(device="cpu")


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
        "--clean_archives",
        type=_str_to_bool,
        default=True,
        help="If True, remove all archives files. defaults to True.",
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
    args = _get_main_download_args()
    _setup_logging(aac_metrics.__package__, args.verbose)

    download_metrics(
        cache_path=args.cache_path,
        tmp_path=args.tmp_path,
        clean_archives=args.clean_archives,
        ptb_tokenizer=args.ptb_tokenizer,
        meteor=args.meteor,
        spice=args.spice,
        fense=args.fense,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    _main_download()
