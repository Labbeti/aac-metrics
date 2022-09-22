#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import os.path as osp
import subprocess
import tempfile
import time

from typing import Any, Hashable, Iterable, Optional


logger = logging.getLogger(__name__)

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


def _ptb_tokenize(
    sentences: Iterable[str],
    audio_ids: Optional[Iterable[Hashable]] = None,
    java_path: str = "java",
    cache_path: str = "$HOME/aac-metrics-cache",
    tmp_path: str = "/tmp",
    verbose: int = 0,
) -> list[list[str]]:
    """Use PTB Tokenizer to process sentences. Should be used only with all the sentences of a subset due to slow computation.

    :param sentences: The sentences to tokenize.
    :param audio_ids: The optional audio names. None will use the audio index as name. defaults to None.
    :param java_path: The path to the java executable. defaults to "java".
    :param cache_path: The path to the external directory containing the JAR program. defaults to "ext".
    :param tmp_path: The path to a temporary directory. defaults to "/tmp".
    :param verbose: The verbose level. defaults to 0.
    :returns: The sentences tokenized.
    """
    cache_path = osp.expandvars(cache_path)

    # Based on https://github.com/audio-captioning/caption-evaluation-tools/blob/c1798df4c91e29fe689b1ccd4ce45439ec966417/coco_caption/pycocoevalcap/tokenizer/ptbtokenizer.py#L30
    sentences = list(sentences)
    if len(sentences) == 0:
        return []

    stanford_fpath = osp.join(cache_path, STANFORD_CORENLP_3_4_1_JAR)

    # Sanity checks
    if not osp.isdir(cache_path):
        raise RuntimeError(f"Cannot find ext directory at {cache_path=}.")
    if not osp.isdir(tmp_path):
        raise RuntimeError(f"Cannot find tmp directory at {tmp_path=}.")
    if not osp.isfile(stanford_fpath):
        raise FileNotFoundError(
            f"Cannot find jar file {STANFORD_CORENLP_3_4_1_JAR} in {cache_path=}."
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
        stderr=subprocess.DEVNULL if verbose < 2 else None,
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
    sents: list[str],
    java_path: str = "java",
    cache_path: str = "$HOME/aac-metrics-cache",
    tmp_path: str = "/tmp",
    verbose: int = 0,
) -> list[str]:
    """Tokenize sentences using PTB Tokenizer then merge them by space.

    Note: PTB tokenizer is a java program that takes a list[str] as input, so calling several times `preprocess_mono_sents` is slow on list[list[str]].
    If you want to process multiple sentences (list[list[str]]), use `preprocess_mult_sents` instead.
    """
    tok_sents = _ptb_tokenize(sents, None, java_path, cache_path, tmp_path, verbose)
    sents = [" ".join(sent) for sent in tok_sents]
    return sents


def preprocess_mult_sents(
    mult_sents: list[list[str]],
    java_path: str = "java",
    cache_path: str = "$HOME/aac-metrics-cache",
    tmp_path: str = "/tmp",
    verbose: int = 0,
) -> list[list[str]]:
    """Tokenize sentences using PTB Tokenizer with only 1 call then merge them by space. """

    # Flat list
    flatten_sents = [sent for sents in mult_sents for sent in sents]
    n_sents_per_item = [len(sents) for sents in mult_sents]

    # Process
    flatten_sents = preprocess_mono_sents(flatten_sents, java_path, cache_path, tmp_path, verbose)

    # Unflat list in the same order
    mult_sents = []
    start = 0
    stop = 0
    for count in n_sents_per_item:
        stop += count
        mult_sents.append(flatten_sents[start:stop])
        start = stop
    return mult_sents
