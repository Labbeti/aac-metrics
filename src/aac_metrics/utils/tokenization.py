#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import os.path as osp
import subprocess
import tempfile
import time

from typing import Any, Hashable, Iterable, TypeVar, Optional

from aac_metrics.utils.misc import _check_java_path


logger = logging.getLogger(__name__)
T = TypeVar("T")


# Path to the stanford corenlp jar
STANFORD_CORENLP_3_4_1_JAR = osp.join("stanford_nlp", "stanford-corenlp-3.4.1.jar")
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
    cache_path: str = "$HOME/aac-metrics-cache",
    java_path: str = "java",
    tmp_path: str = "/tmp",
    verbose: int = 0,
) -> list[list[str]]:
    """Use PTB Tokenizer to process sentences. Should be used only with all the sentences of a subset due to slow computation.

    :param sentences: The sentences to tokenize.
    :param audio_ids: The optional audio names. None will use the audio index as name. defaults to None.
    :param cache_path: The path to the external directory containing the JAR program. defaults to "$HOME/aac-metrics-cache".
    :param java_path: The path to the java executable. defaults to "java".
    :param tmp_path: The path to a temporary directory. defaults to "/tmp".
    :param verbose: The verbose level. defaults to 0.
    :returns: The sentences tokenized.
    """
    cache_path = osp.expandvars(cache_path)
    java_path = osp.expandvars(java_path)
    tmp_path = osp.expandvars(tmp_path)

    # Based on https://github.com/audio-captioning/caption-evaluation-tools/blob/c1798df4c91e29fe689b1ccd4ce45439ec966417/coco_caption/pycocoevalcap/tokenizer/ptbtokenizer.py#L30
    sentences = list(sentences)
    if len(sentences) == 0:
        return []

    stanford_fpath = osp.join(cache_path, STANFORD_CORENLP_3_4_1_JAR)

    # Sanity checks
    if not osp.isdir(cache_path):
        raise RuntimeError(f"Cannot find cache directory at {cache_path=}.")
    if not osp.isdir(tmp_path):
        raise RuntimeError(f"Cannot find tmp directory at {tmp_path=}.")
    if not osp.isfile(stanford_fpath):
        raise FileNotFoundError(
            f"Cannot find jar file {STANFORD_CORENLP_3_4_1_JAR} in {cache_path=}. Maybe run 'aac-metrics-download' before or specify a 'cache_path' directory."
        )
    if not _check_java_path(java_path):
        raise ValueError(
            f"Cannot find java executable with {java_path=} to tokenize sentences."
        )

    start_time = time.perf_counter()
    if verbose >= 2:
        logger.debug(
            f"Start executing {STANFORD_CORENLP_3_4_1_JAR} JAR file for tokenization. ({len(sentences)=})"
        )

    cmd = [
        java_path,
        "-cp",
        stanford_fpath,
        "edu.stanford.nlp.process.PTBTokenizer",
        "-preserveLines",
        "-lowerCase",
    ]

    # ======================================================
    # prepare data for PTB AACTokenizer
    # ======================================================
    if audio_ids is None:
        audio_ids = list(range(len(sentences)))
    else:
        audio_ids = list(audio_ids)

    if len(audio_ids) != len(sentences):
        raise ValueError(
            f"Invalid number of audio ids ({len(audio_ids)}) with sentences len={len(sentences)}."
        )

    sentences = "\n".join(sentences)

    # ======================================================
    # save sentences to temporary file
    # ======================================================
    tmp_file = tempfile.NamedTemporaryFile(
        delete=False,
        dir=tmp_path,
        prefix="ptb_sentences_",
        suffix=".txt",
    )
    tmp_file.write(sentences.encode())
    tmp_file.close()

    # ======================================================
    # tokenize sentence
    # ======================================================
    cmd.append(osp.basename(tmp_file.name))
    p_tokenizer = subprocess.Popen(
        cmd,
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL if verbose <= 2 else None,
    )
    token_lines = p_tokenizer.communicate(input=sentences.rstrip().encode())[0]
    token_lines = token_lines.decode()
    lines = token_lines.split("\n")
    # remove temp file
    os.remove(tmp_file.name)

    # ======================================================
    # create dictionary for tokenized captions
    # ======================================================
    outs: Any = [None for _ in range(len(lines))]
    if len(audio_ids) != len(lines):
        raise RuntimeError(
            f"PTB tokenize error: expected {len(audio_ids)} lines in output file but found {len(lines)}."
        )

    for k, line in zip(audio_ids, lines):
        tokenized_caption = [
            w for w in line.rstrip().split(" ") if w not in PTB_PUNCTUATIONS
        ]
        outs[k] = tokenized_caption
    assert all(out is not None for out in outs)

    if verbose >= 2:
        duration = time.perf_counter() - start_time
        logger.debug(f"Tokenization finished in {duration:.2f}s.")

    return outs


def preprocess_mono_sents(
    sentences: list[str],
    cache_path: str = "$HOME/aac-metrics-cache",
    java_path: str = "java",
    tmp_path: str = "/tmp",
    verbose: int = 0,
) -> list[str]:
    """Tokenize sentences using PTB Tokenizer then merge them by space.

    Note: PTB tokenizer is a java program that takes a list[str] as input, so calling several times `preprocess_mono_sents` is slow on list[list[str]].
    If you want to process multiple sentences (list[list[str]]), use `preprocess_mult_sents` instead.

    :param sentences: The list of sentences to process.
    :param cache_path: The path to the external code directory. defaults to "$HOME/aac-metrics-cache".
    :param java_path: The path to the java executable. defaults to "java".
    :param tmp_path: Temporary directory path. defaults to "/tmp".
    :returns: The sentences processed by the tokenizer.
    """
    tok_sents = ptb_tokenize_batch(
        sentences, None, cache_path, java_path, tmp_path, verbose
    )
    sentences = [" ".join(sent) for sent in tok_sents]
    return sentences


def preprocess_mult_sents(
    mult_sentences: list[list[str]],
    cache_path: str = "$HOME/aac-metrics-cache",
    java_path: str = "java",
    tmp_path: str = "/tmp",
    verbose: int = 0,
) -> list[list[str]]:
    """Tokenize multiple sentences using PTB Tokenizer with only 1 call then merge them by space.

    :param mult_sentences: The list of list of sentences to process.
    :param cache_path: The path to the external code directory. defaults to "$HOME/aac-metrics-cache".
    :param java_path: The path to the java executable. defaults to "java".
    :param tmp_path: Temporary directory path. defaults to "/tmp".
    :returns: The multiple sentences processed by the tokenizer.
    """

    # Flat list
    flatten_sents, sizes = flat_list(mult_sentences)
    flatten_sents = preprocess_mono_sents(
        flatten_sents,
        cache_path,
        java_path,
        tmp_path,
        verbose,
    )
    mult_sentences = unflat_list(flatten_sents, sizes)
    return mult_sentences


def flat_list(lst: list[list[T]]) -> tuple[list[T], list[int]]:
    flatten_lst = [element for sublst in lst for element in sublst]
    sizes = [len(sents) for sents in lst]
    return flatten_lst, sizes


def unflat_list(flatten_lst: list[T], sizes: list[int]) -> list[list[T]]:
    lst = []
    start = 0
    stop = 0
    for count in sizes:
        stop += count
        lst.append(flatten_lst[start:stop])
        start = stop
    return lst
