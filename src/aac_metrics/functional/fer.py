#!/usr/bin/env python
# -*- coding: utf-8 -*-

import hashlib
import logging
import os
import re
import requests

from collections import namedtuple
from os import environ, makedirs
from os.path import exists, expanduser, join
from typing import Mapping, Optional, Union

import numpy as np
import torch
import transformers

from torch import nn, Tensor
from tqdm import tqdm
from transformers import logging as tfmers_logging
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from aac_metrics.utils.checks import is_mono_sents
from aac_metrics.utils.globals import _get_device


DEFAULT_FER_MODEL = "echecker_clotho_audiocaps_base"


_DEFAULT_PROXIES = {
    "http": "socks5h://127.0.0.1:1080",
    "https": "socks5h://127.0.0.1:1080",
}
_PRETRAIN_ECHECKERS_DICT = {
    "echecker_clotho_audiocaps_base": (
        "https://github.com/blmoistawinde/fense/releases/download/V0.1/echecker_clotho_audiocaps_base.ckpt",
        "1a719f090af70614bbdb9f9437530b7e133c48cfa4a58d964de0d47fc974a2fa",
    ),
    "echecker_clotho_audiocaps_tiny": (
        "https://github.com/blmoistawinde/fense/releases/download/V0.1/echecker_clotho_audiocaps_tiny.ckpt",
        "90ed0ac5033ec497ec66d4f68588053813e085671136dae312097c96c504f673",
    ),
}
_ERROR_NAMES = (
    "add_tail",
    "repeat_event",
    "repeat_adv",
    "remove_conj",
    "remove_verb",
    "error",
)

_RemoteFileMetadata = namedtuple("RemoteFileMetadata", ["filename", "url", "checksum"])

pylog = logging.getLogger(__name__)


class BERTFlatClassifier(nn.Module):
    def __init__(self, model_type: str, num_classes: int = 5) -> None:
        super().__init__()
        self.model_type = model_type
        self.num_classes = num_classes
        self.encoder = AutoModel.from_pretrained(model_type)
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
        self.clf = nn.Linear(self.encoder.config.hidden_size, num_classes)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = DEFAULT_FER_MODEL,
        device: Union[str, torch.device, None] = "cuda_if_available",
        use_proxy: bool = False,
        proxies: Optional[dict[str, str]] = None,
        verbose: int = 0,
    ) -> "BERTFlatClassifier":
        return __load_pretrain_echecker(
            echecker_model=model_name,
            device=device,
            use_proxy=use_proxy,
            proxies=proxies,
            verbose=verbose,
        )

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        x = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(x)
        logits = self.clf(x)
        return logits


def fer(
    candidates: list[str],
    return_all_scores: bool = True,
    echecker: Union[str, BERTFlatClassifier] = DEFAULT_FER_MODEL,
    echecker_tokenizer: Optional[AutoTokenizer] = None,
    error_threshold: float = 0.9,
    device: Union[str, torch.device, None] = "cuda_if_available",
    batch_size: int = 32,
    reset_state: bool = True,
    return_probs: bool = False,
    verbose: int = 0,
) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Tensor]]]:
    """Return Fluency Error Rate (FER) detected by a pre-trained BERT model.

    - Paper: https://arxiv.org/abs/2110.04684
    - Original implementation: https://github.com/blmoistawinde/fense

    :param candidates: The list of sentences to evaluate.
    :param mult_references: The list of list of sentences used as target.
    :param return_all_scores: If True, returns a tuple containing the globals and locals scores.
        Otherwise returns a scalar tensor containing the main global score.
        defaults to True.
    :param echecker: The echecker model used to detect fluency errors.
        Can be "echecker_clotho_audiocaps_base", "echecker_clotho_audiocaps_tiny", "none" or None.
        defaults to "echecker_clotho_audiocaps_base".
    :param echecker_tokenizer: The tokenizer of the echecker model.
        If None and echecker is not None, this value will be inferred with `echecker.model_type`.
        defaults to None.
    :param error_threshold: The threshold used to detect fluency errors for echecker model. defaults to 0.9.
    :param device: The PyTorch device used to run FENSE models. If "cuda_if_available", it will use cuda if available. defaults to "cuda_if_available".
    :param batch_size: The batch size of the echecker models. defaults to 32.
    :param reset_state: If True, reset the state of the PyTorch global generator after the initialization of the pre-trained models. defaults to True.
    :param return_probs: If True, return each individual error probability given by the fluency detector model. defaults to False.
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple of globals and locals scores or a scalar tensor with the main global score.
    """
    if not is_mono_sents(candidates):
        error_msg = f"Invalid candidates type. (expected list[str], found {candidates.__class__.__name__})"
        raise ValueError(error_msg)

    # Init models
    echecker, echecker_tokenizer = _load_echecker_and_tokenizer(
        echecker=echecker,
        echecker_tokenizer=echecker_tokenizer,
        device=device,
        reset_state=reset_state,
        verbose=verbose,
    )

    # Compute and apply fluency error detection penalty
    probs_outs_sents = __detect_error_sents(
        echecker=echecker,
        echecker_tokenizer=echecker_tokenizer,
        sents=candidates,
        batch_size=batch_size,
        device=device,
    )
    fer_scores = (probs_outs_sents["error"] > error_threshold).astype(float)

    fer_scores = torch.from_numpy(fer_scores)
    fer_score = fer_scores.mean()

    if return_all_scores:
        fer_outs_corpus = {
            "fer": fer_score,
        }
        fer_outs_sents = {
            "fer": fer_scores,
        }

        if return_probs:
            probs_outs_sents = {f"fer.{k}_prob": v for k, v in probs_outs_sents.items()}
            probs_outs_sents = {
                k: torch.from_numpy(v) for k, v in probs_outs_sents.items()
            }
            probs_outs_corpus = {k: v.mean() for k, v in probs_outs_sents.items()}

            fer_outs_corpus = probs_outs_corpus | fer_outs_corpus
            fer_outs_sents = probs_outs_sents | fer_outs_sents

        fer_outs = fer_outs_corpus, fer_outs_sents

        return fer_outs
    else:
        return fer_score


def _use_new_echecker_loading() -> bool:
    version = transformers.__version__
    major, minor, _patch = map(int, version.split("."))
    return major > 4 or (major == 4 and minor >= 31)


# - Private functions
def _load_echecker_and_tokenizer(
    echecker: Union[str, BERTFlatClassifier] = DEFAULT_FER_MODEL,
    echecker_tokenizer: Optional[AutoTokenizer] = None,
    device: Union[str, torch.device, None] = "cuda_if_available",
    reset_state: bool = True,
    verbose: int = 0,
) -> tuple[BERTFlatClassifier, AutoTokenizer]:
    state = torch.random.get_rng_state()

    device = _get_device(device)
    if isinstance(echecker, str):
        echecker = __load_pretrain_echecker(
            echecker_model=echecker, device=device, verbose=verbose
        )

    if echecker_tokenizer is None:
        echecker_tokenizer = AutoTokenizer.from_pretrained(echecker.model_type)  # type: ignore

    echecker = echecker.eval()
    for p in echecker.parameters():
        p.requires_grad_(False)

    if reset_state:
        torch.random.set_rng_state(state)

    return echecker, echecker_tokenizer  # type: ignore


def __detect_error_sents(
    echecker: BERTFlatClassifier,
    echecker_tokenizer: PreTrainedTokenizerFast,
    sents: list[str],
    batch_size: int,
    device: Union[str, torch.device, None],
    max_len: int = 64,
) -> dict[str, np.ndarray]:
    device = _get_device(device)

    if len(sents) <= batch_size:
        batch = __infer_preprocess(
            echecker_tokenizer,
            sents,
            max_len=max_len,
            device=device,
            dtype=torch.long,
        )
        logits: Tensor = echecker(**batch)
        assert not logits.requires_grad
        # batch_logits: (bsize, num_classes=6)
        # note: fix error in the original fense code: https://github.com/blmoistawinde/fense/blob/main/fense/evaluator.py#L69
        probs = logits.sigmoid().transpose(0, 1).cpu().numpy()
        probs_dic: dict[str, np.ndarray] = dict(zip(_ERROR_NAMES, probs))

    else:
        dic_lst_probs = {name: [] for name in _ERROR_NAMES}

        for i in range(0, len(sents), batch_size):
            batch = __infer_preprocess(
                echecker_tokenizer,
                sents[i : i + batch_size],
                max_len=max_len,
                device=device,
                dtype=torch.long,
            )

            batch_logits: Tensor = echecker(**batch)
            assert not batch_logits.requires_grad
            # batch_logits: (bsize, num_classes=6)
            # classes: add_tail, repeat_event, repeat_adv, remove_conj, remove_verb, error
            probs = batch_logits.sigmoid().cpu().numpy()

            for j, name in enumerate(dic_lst_probs.keys()):
                dic_lst_probs[name].append(probs[:, j])

        probs_dic = {
            name: np.concatenate(probs) for name, probs in dic_lst_probs.items()
        }

    return probs_dic


def __check_download_resource(
    remote: _RemoteFileMetadata,
    use_proxy: bool = False,
    proxies: Optional[dict[str, str]] = None,
) -> str:
    data_home = __get_data_home()
    file_path = os.path.join(data_home, remote.filename)
    if not os.path.exists(file_path):
        # currently don't capture error at this level, assume download success
        file_path = __download(remote, data_home, use_proxy, proxies)
    return file_path


def __infer_preprocess(
    tokenizer: PreTrainedTokenizerFast,
    texts: list[str],
    max_len: int,
    device: Union[str, torch.device, None],
    dtype: torch.dtype,
) -> Mapping[str, Tensor]:
    device = _get_device(device)
    texts = __text_preprocess(texts)  # type: ignore
    batch = tokenizer(texts, truncation=True, padding="max_length", max_length=max_len)
    for k in ("input_ids", "attention_mask", "token_type_ids"):
        batch[k] = torch.as_tensor(batch[k], device=device, dtype=dtype)  # type: ignore
    return batch


def __download(
    remote: _RemoteFileMetadata,
    file_path: Optional[str] = None,
    use_proxy: bool = False,
    proxies: Optional[dict[str, str]] = None,
) -> str:
    data_home = __get_data_home()
    file_path = __fetch_remote(remote, data_home, use_proxy, proxies)
    return file_path


def __download_with_bar(
    url: str,
    file_path: str,
    use_proxy: bool = False,
    proxies: Optional[dict[str, str]] = None,
) -> str:
    if use_proxy and proxies is None:
        proxies = _DEFAULT_PROXIES

    # Streaming, so we can iterate over the response.
    response = requests.get(url, stream=True, proxies=proxies)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 KB
    progress_bar = tqdm(total=total_size_in_bytes, unit="B", unit_scale=True)
    with open(file_path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        raise Exception("ERROR, something went wrong with the downloading")
    return file_path


def __fetch_remote(
    remote: _RemoteFileMetadata,
    dirname: Optional[str] = None,
    use_proxy: bool = False,
    proxies: Optional[dict[str, str]] = None,
) -> str:
    file_path = remote.filename if dirname is None else join(dirname, remote.filename)
    file_path = __download_with_bar(remote.url, file_path, use_proxy, proxies)
    checksum = __sha256(file_path)
    if remote.checksum != checksum:
        raise IOError(
            "{} has an SHA256 checksum ({}) "
            "differing from expected ({}), "
            "file may be corrupted.".format(file_path, checksum, remote.checksum)
        )
    return file_path


def __get_data_home(data_home: Optional[str] = None) -> str:
    if data_home is None:
        DEFAULT_DATA_HOME = join(torch.hub.get_dir(), "fense_data")
        data_home = environ.get("FENSE_DATA", DEFAULT_DATA_HOME)

    data_home: str
    data_home = expanduser(data_home)
    if not exists(data_home):
        makedirs(data_home)
    return data_home


def __load_pretrain_echecker(
    echecker_model: str,
    device: Union[str, torch.device, None] = "cuda_if_available",
    use_proxy: bool = False,
    proxies: Optional[dict[str, str]] = None,
    verbose: int = 0,
) -> BERTFlatClassifier:
    if echecker_model not in _PRETRAIN_ECHECKERS_DICT:
        raise ValueError(
            f"Invalid argument {echecker_model=}. (expected one of {tuple(_PRETRAIN_ECHECKERS_DICT.keys())})"
        )

    device = _get_device(device)
    tfmers_logging.set_verbosity_error()  # suppress loading warnings
    url, checksum = _PRETRAIN_ECHECKERS_DICT[echecker_model]
    remote = _RemoteFileMetadata(
        filename=f"{echecker_model}.ckpt", url=url, checksum=checksum
    )
    file_path = __check_download_resource(remote, use_proxy, proxies)

    if verbose >= 2:
        pylog.debug(f"Loading echecker model from '{file_path}'.")

    model_states = torch.load(file_path)
    model_type = model_states["model_type"]
    num_classes = model_states["num_classes"]
    state_dict = model_states["state_dict"]

    if verbose >= 2:
        pylog.debug(
            f"Loading echecker model type '{model_type}' with '{num_classes}' classes."
        )

    echecker = BERTFlatClassifier(
        model_type=model_type,
        num_classes=num_classes,
    )

    # To support transformers > 4.31, because this lib changed BertEmbedding state_dict
    if _use_new_echecker_loading():
        state_dict.pop("encoder.embeddings.position_ids")

    echecker.load_state_dict(state_dict)
    echecker.eval()
    echecker.to(device=device)
    return echecker


def __sha256(path: str) -> str:
    """Calculate the sha256 hash of the file at path."""
    sha256hash = hashlib.sha256()
    chunk_size = 8192
    with open(path, "rb") as f:
        while True:
            buffer = f.read(chunk_size)
            if not buffer:
                break
            sha256hash.update(buffer)
    return sha256hash.hexdigest()


def __text_preprocess(inp: Union[str, list[str]]) -> Union[str, list[str]]:
    if isinstance(inp, str):
        return re.sub(r"[^\w\s]", "", inp).lower()
    else:
        return [re.sub(r"[^\w\s]", "", x).lower() for x in inp]
