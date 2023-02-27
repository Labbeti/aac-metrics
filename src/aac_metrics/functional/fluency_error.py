#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BASED ON https://github.com/blmoistawinde/fense/
"""

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

from torch import nn, Tensor
from tqdm import tqdm
from transformers import logging as tfmers_logging
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast


# config according to the settings on your computer, this should be default setting of shadowsocks
DEFAULT_PROXIES = {
    "http": "socks5h://127.0.0.1:1080",
    "https": "socks5h://127.0.0.1:1080",
}
PRETRAIN_ECHECKERS_DICT = {
    "echecker_clotho_audiocaps_base": (
        "https://github.com/blmoistawinde/fense/releases/download/V0.1/echecker_clotho_audiocaps_base.ckpt",
        "1a719f090af70614bbdb9f9437530b7e133c48cfa4a58d964de0d47fc974a2fa",
    ),
    "echecker_clotho_audiocaps_tiny": (
        "https://github.com/blmoistawinde/fense/releases/download/V0.1/echecker_clotho_audiocaps_tiny.ckpt",
        "90ed0ac5033ec497ec66d4f68588053813e085671136dae312097c96c504f673",
    ),
    "none": (None, None),
}

RemoteFileMetadata = namedtuple("RemoteFileMetadata", ["filename", "url", "checksum"])

logger = logging.getLogger(__name__)

ERROR_NAMES = (
    "add_tail",
    "repeat_event",
    "repeat_adv",
    "remove_conj",
    "remove_verb",
    "error",
)


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
        model_name: str = "echecker_clotho_audiocaps_base",
        device: Union[str, torch.device, None] = "auto",
        use_proxy: bool = False,
        proxies: Optional[dict[str, str]] = None,
        verbose: int = 0,
    ) -> "BERTFlatClassifier":
        return __load_pretrain_echecker(model_name, device, use_proxy, proxies, verbose)

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


def fluency_error(
    candidates: list[str],
    return_all_scores: bool = True,
    echecker: Union[str, BERTFlatClassifier] = "echecker_clotho_audiocaps_base",
    echecker_tokenizer: Optional[AutoTokenizer] = None,
    error_threshold: float = 0.9,
    device: Union[str, torch.device, None] = "auto",
    batch_size: int = 32,
    reset_state: bool = True,
    verbose: int = 0,
) -> Union[Tensor, tuple[dict[str, Tensor], dict[str, Tensor]]]:
    """Return fluency error detected by a pre-trained BERT model.

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
    :param device: The PyTorch device used to run FENSE models. If "auto", it will use cuda if available. defaults to "cpu".
    :param batch_size: The batch size of the echecker models. defaults to 32.
    :param verbose: The verbose level. defaults to 0.
    :returns: A tuple of globals and locals scores or a scalar tensor with the main global score.
    """

    # Init models
    echecker, echecker_tokenizer = _load_echecker_and_tokenizer(
        echecker, echecker_tokenizer, device, reset_state, verbose
    )

    # Compute and apply fluency error detection penalty
    sents_probs_dic = __detect_error_sents(
        echecker,
        echecker_tokenizer,  # type: ignore
        candidates,
        batch_size,
        device,
    )
    fluency_errors = (sents_probs_dic["error"] > error_threshold).astype(float)
    sents_probs_dic = {f"fluerr.{k}_prob": v for k, v in sents_probs_dic.items()}

    sents_probs_dic = {k: torch.from_numpy(v) for k, v in sents_probs_dic.items()}
    corpus_probs_dic = {k: v.mean() for k, v in sents_probs_dic.items()}

    fluency_errors = torch.from_numpy(fluency_errors)
    fluency_error = fluency_errors.mean()

    if return_all_scores:
        sents_scores = {
            "fluency_error": fluency_errors,
        } | sents_probs_dic
        corpus_scores = {
            "fluency_error": fluency_error,
        } | corpus_probs_dic

        return corpus_scores, sents_scores
    else:
        return fluency_error


# - Private functions
def _load_echecker_and_tokenizer(
    echecker: Union[str, BERTFlatClassifier] = "echecker_clotho_audiocaps_base",
    echecker_tokenizer: Optional[AutoTokenizer] = None,
    device: Union[str, torch.device, None] = "auto",
    reset_state: bool = True,
    verbose: int = 0,
) -> tuple[BERTFlatClassifier, AutoTokenizer]:
    state = torch.random.get_rng_state()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, str):
        device = torch.device(device)

    if isinstance(echecker, str):
        echecker = __load_pretrain_echecker(echecker, device, verbose=verbose)

    if echecker_tokenizer is None:
        echecker_tokenizer = AutoTokenizer.from_pretrained(echecker.model_type)

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
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, str):
        device = torch.device(device)

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
        probs_dic = dict(zip(ERROR_NAMES, probs))

    else:
        probs_dic = {name: [] for name in ERROR_NAMES}

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

            for j, name in enumerate(probs_dic.keys()):
                probs_dic[name].append(probs[:, j])

        probs_dic = {name: np.concatenate(probs) for name, probs in probs_dic.items()}

    return probs_dic


def __check_download_resource(
    remote: RemoteFileMetadata,
    use_proxy: bool = False,
    proxies: Optional[dict[str, str]] = None,
) -> str:
    proxies = DEFAULT_PROXIES if use_proxy and proxies is None else proxies
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
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, str):
        device = torch.device(device)

    texts = __text_preprocess(texts)  # type: ignore
    batch = tokenizer(texts, truncation=True, padding="max_length", max_length=max_len)
    for k in ("input_ids", "attention_mask", "token_type_ids"):
        batch[k] = torch.as_tensor(batch[k], device=device, dtype=dtype)  # type: ignore
    return batch


def __download(
    remote: RemoteFileMetadata,
    file_path: Optional[str] = None,
    use_proxy: bool = False,
    proxies: Optional[dict[str, str]] = DEFAULT_PROXIES,
) -> str:
    data_home = __get_data_home()
    file_path = __fetch_remote(remote, data_home, use_proxy, proxies)
    return file_path


def __download_with_bar(
    url: str,
    file_path: str,
    proxies: Optional[dict[str, str]] = DEFAULT_PROXIES,
) -> str:
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
    remote: RemoteFileMetadata,
    dirname: Optional[str] = None,
    use_proxy: bool = False,
    proxies: Optional[dict[str, str]] = DEFAULT_PROXIES,
) -> str:
    """Helper function to download a remote dataset into path
    Fetch a dataset pointed by remote's url, save into path using remote's
    filename and ensure its integrity based on the SHA256 Checksum of the
    downloaded file.
    Parameters
    ----------
    remote : RemoteFileMetadata
        Named tuple containing remote dataset meta information: url, filename
        and checksum
    dirname : string
        Directory to save the file to.
    Returns
    -------
    file_path: string
        Full path of the created file.
    """

    file_path = remote.filename if dirname is None else join(dirname, remote.filename)
    proxies = None if not use_proxy else proxies
    file_path = __download_with_bar(remote.url, file_path, proxies)
    checksum = __sha256(file_path)
    if remote.checksum != checksum:
        raise IOError(
            "{} has an SHA256 checksum ({}) "
            "differing from expected ({}), "
            "file may be corrupted.".format(file_path, checksum, remote.checksum)
        )
    return file_path


def __get_data_home(data_home: Optional[str] = None) -> str:  # type: ignore
    """Return the path of the scikit-learn data dir.
    This folder is used by some large dataset loaders to avoid downloading the
    data several times.
    By default the data dir is set to a folder named 'fense_data' in the
    user home folder.
    Alternatively, it can be set by the 'FENSE_DATA' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.
    If the folder does not already exist, it is automatically created.
    Parameters
    ----------
    data_home : str | None
        The path to data dir.
    """
    if data_home is None:
        data_home = environ.get("FENSE_DATA", join(torch.hub.get_dir(), "fense_data"))

    data_home: str
    data_home = expanduser(data_home)
    if not exists(data_home):
        makedirs(data_home)
    return data_home


def __load_pretrain_echecker(
    echecker_model: str,
    device: Union[str, torch.device, None] = "auto",
    use_proxy: bool = False,
    proxies: Optional[dict[str, str]] = None,
    verbose: int = 0,
) -> BERTFlatClassifier:
    if echecker_model not in PRETRAIN_ECHECKERS_DICT:
        raise ValueError(
            f"Invalid argument {echecker_model=}. (expected one of {tuple(PRETRAIN_ECHECKERS_DICT.keys())})"
        )

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, str):
        device = torch.device(device)

    tfmers_logging.set_verbosity_error()  # suppress loading warnings
    url, checksum = PRETRAIN_ECHECKERS_DICT[echecker_model]
    remote = RemoteFileMetadata(
        filename=f"{echecker_model}.ckpt", url=url, checksum=checksum
    )
    file_path = __check_download_resource(remote, use_proxy, proxies)

    if verbose >= 2:
        logger.debug(f"Loading echecker model from '{file_path}'.")

    model_states = torch.load(file_path)
    echecker = BERTFlatClassifier(
        model_type=model_states["model_type"],
        num_classes=model_states["num_classes"],
    )
    echecker.load_state_dict(model_states["state_dict"])
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
