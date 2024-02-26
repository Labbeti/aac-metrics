#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import os.path as osp
import subprocess
import tempfile
import time

from pathlib import Path
from typing import Any, Hashable, Iterable, Optional, Union

from aac_metrics.utils.checks import check_java_path, is_mono_sents
from aac_metrics.utils.collections import flat_list, unflat_list
from aac_metrics.utils.globals import (
    _get_cache_path,
    _get_java_path,
    _get_tmp_path,
)


pylog = logging.getLogger(__name__)


# Path to the stanford corenlp jar
FNAME_STANFORD_CORENLP_3_4_1_JAR = osp.join(
    "aac-metrics",
    "stanford_nlp",
    "stanford-corenlp-3.4.1.jar",
)
# Punctuations to be removed from the sentences
PTB_PUNCTUATIONS = (
    "''",
    "'",
    "``",
    "`",
    "-LRB-",
    "-RRB-",
    "-LCB-",
    "-RCB-",
    ".",
    "?",
    "!",
    ",",
    ":",
    "-",
    "--",
    "...",
    ";",
)


def ptb_tokenize_batch(
    sentences: Iterable[str],
    audio_ids: Optional[Iterable[Hashable]] = None,
    cache_path: Union[str, Path, None] = None,
    java_path: Union[str, Path, None] = None,
    tmp_path: Union[str, Path, None] = None,
    punctuations: Iterable[str] = PTB_PUNCTUATIONS,
    normalize_apostrophe: bool = False,
    verbose: int = 0,
) -> list[list[str]]:
    """Use PTB Tokenizer to process sentences. Should be used only with all the sentences of a subset due to slow computation.

    :param sentences: The sentences to tokenize.
    :param audio_ids: The optional audio names for the PTB Tokenizer program. None will use the audio index as name. defaults to None.
    :param cache_path: The path to the external directory containing the JAR program. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_cache_path`.
    :param java_path: The path to the java executable. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_java_path`.
    :param tmp_path: The path to a temporary directory. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_tmp_path`.
    :param normalize_apostrophe: If True, add apostrophes for French language. defaults to False.
    :param verbose: The verbose level. defaults to 0.
    :returns: The sentences tokenized as list[list[str]].
    """
    # Originally based on https://github.com/audio-captioning/caption-evaluation-tools/blob/c1798df4c91e29fe689b1ccd4ce45439ec966417/caption/pycocoevalcap/tokenizer/ptbtokenizer.py#L30

    sentences = list(sentences)

    if not is_mono_sents(sentences):
        raise ValueError("Invalid argument sentences. (not a list[str] of sentences)")

    if len(sentences) == 0:
        return []

    cache_path = _get_cache_path(cache_path)
    java_path = _get_java_path(java_path)
    tmp_path = _get_tmp_path(tmp_path)
    punctuations = list(punctuations)

    stanford_fpath = osp.join(cache_path, FNAME_STANFORD_CORENLP_3_4_1_JAR)

    # Sanity checks
    if __debug__:
        newlines_count = sum(sent.count("\n") for sent in sentences)
        if newlines_count > 0:
            raise ValueError(
                f"Invalid argument sentences for tokenization. (found {newlines_count} newlines character '\\n')"
            )

        if not osp.isdir(cache_path):
            raise RuntimeError(f"Cannot find cache directory at {cache_path=}.")
        if not osp.isdir(tmp_path):
            raise RuntimeError(f"Cannot find tmp directory at {tmp_path=}.")
        if not osp.isfile(stanford_fpath):
            raise FileNotFoundError(
                f"Cannot find JAR file '{stanford_fpath}' for tokenization. Maybe run 'aac-metrics-download' or specify another 'cache_path' directory."
            )
        if not check_java_path(java_path):
            raise RuntimeError(
                f"Invalid Java executable to tokenize sentences. ({java_path})"
            )

    start_time = time.perf_counter()
    if verbose >= 2:
        pylog.debug(
            f"Start executing {FNAME_STANFORD_CORENLP_3_4_1_JAR} JAR file for tokenization. ({len(sentences)=})"
        )

    cmd = [
        java_path,
        "-cp",
        stanford_fpath,
        "edu.stanford.nlp.process.PTBTokenizer",
        "-preserveLines",
        "-lowerCase",
    ]

    if audio_ids is None:
        audio_ids = list(range(len(sentences)))
    else:
        audio_ids = list(audio_ids)

    if len(audio_ids) != len(sentences):
        raise ValueError(
            f"Invalid number of audio ids ({len(audio_ids)}) with sentences len={len(sentences)}."
        )

    sentences = "\n".join(sentences)
    if normalize_apostrophe:
        replaces = {
            " s ": " s'",
            "'": "' ",
            "'  ": "' ",
            " '": "'",
        }
        for old, new in replaces.items():
            sentences = sentences.replace(old, new)

    tmp_file = tempfile.NamedTemporaryFile(
        delete=False,
        dir=tmp_path,
        prefix="ptb_sentences_",
        suffix=".txt",
    )
    tmp_file.write(sentences.encode())
    tmp_file.close()

    cmd.append(osp.basename(tmp_file.name))
    p_tokenizer = subprocess.Popen(
        cmd,
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL if verbose <= 2 else None,
    )
    encoded_sentences = sentences.rstrip().encode()
    token_lines = p_tokenizer.communicate(input=encoded_sentences)[0]
    token_lines = token_lines.decode()
    lines = token_lines.split("\n")
    # remove temp file
    os.remove(tmp_file.name)

    if len(audio_ids) != len(lines):
        raise RuntimeError(
            f"PTB tokenize error: expected {len(audio_ids)} lines in output file but found {len(lines)}."
            f"Maybe check if there is any newline character '\\n' in your sentences or disable preprocessing tokenization."
        )

    outs: Any = [None for _ in range(len(lines))]
    for k, line in zip(audio_ids, lines):
        tokenized_caption = [
            w for w in line.rstrip().split(" ") if w not in punctuations
        ]
        outs[k] = tokenized_caption
    assert all(
        out is not None for out in outs
    ), "INTERNAL ERROR: PTB tokenizer output is invalid."

    if verbose >= 2:
        duration = time.perf_counter() - start_time
        pylog.debug(f"Tokenization finished in {duration:.2f}s.")

    return outs


def preprocess_mono_sents(
    sentences: list[str],
    cache_path: Union[str, Path, None] = None,
    java_path: Union[str, Path, None] = None,
    tmp_path: Union[str, Path, None] = None,
    punctuations: Iterable[str] = PTB_PUNCTUATIONS,
    normalize_apostrophe: bool = False,
    verbose: int = 0,
) -> list[str]:
    """Tokenize sentences using PTB Tokenizer then merge them by space.

    .. warning::
        PTB tokenizer is a java program that takes a list[str] as input, so calling several times this function is slow on list[list[str]].

        If you want to process multiple sentences (list[list[str]]), use :func:`~aac_metrics.utils.tokenization.preprocess_mult_sents` instead.

    :param sentences: The list of sentences to process.
    :param cache_path: The path to the external code directory. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_cache_path`.
    :param java_path: The path to the java executable. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_java_path`.
    :param tmp_path: Temporary directory path. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_tmp_path`.
    :param normalize_apostrophe: If True, add apostrophes for French language. defaults to False.
    :param verbose: The verbose level. defaults to 0.
    :returns: The sentences processed by the tokenizer.
    """
    tok_sents = ptb_tokenize_batch(
        sentences=sentences,
        audio_ids=None,
        cache_path=cache_path,
        java_path=java_path,
        tmp_path=tmp_path,
        punctuations=punctuations,
        normalize_apostrophe=normalize_apostrophe,
        verbose=verbose,
    )
    sentences = [" ".join(sent) for sent in tok_sents]
    return sentences


def preprocess_mult_sents(
    mult_sentences: list[list[str]],
    cache_path: Union[str, Path, None] = None,
    java_path: Union[str, Path, None] = None,
    tmp_path: Union[str, Path, None] = None,
    punctuations: Iterable[str] = PTB_PUNCTUATIONS,
    normalize_apostrophe: bool = False,
    verbose: int = 0,
) -> list[list[str]]:
    """Tokenize multiple sentences using PTB Tokenizer with only one call then merge them by space.

    :param mult_sentences: The list of list of sentences to process.
    :param cache_path: The path to the external code directory. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_cache_path`.
    :param java_path: The path to the java executable. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_java_path`.
    :param tmp_path: Temporary directory path. defaults to the value returned by :func:`~aac_metrics.utils.paths.get_default_tmp_path`.
    :param normalize_apostrophe: If True, add apostrophes for French language. defaults to False.
    :param verbose: The verbose level. defaults to 0.
    :returns: The multiple sentences processed by the tokenizer.
    """
    flatten_sents, sizes = flat_list(mult_sentences)
    flatten_sents = preprocess_mono_sents(
        sentences=flatten_sents,
        cache_path=cache_path,
        java_path=java_path,
        tmp_path=tmp_path,
        punctuations=punctuations,
        normalize_apostrophe=normalize_apostrophe,
        verbose=verbose,
    )
    mult_sentences = unflat_list(flatten_sents, sizes)
    return mult_sentences
