#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BASED ON https://github.com/blmoistawinde/fense/
"""

import hashlib
import os
import re
import requests
import shutil

from collections import namedtuple
from os import environ, makedirs
from os.path import exists, expanduser, join
from typing import Callable, Optional, Union

import torch

from torch import nn, Tensor
from tqdm import tqdm
from transformers.models.auto.modeling_auto import AutoModel


RemoteFileMetadata = namedtuple("RemoteFileMetadata", ["filename", "url", "checksum"])

# config according to the settings on your computer, this should be default setting of shadowsocks
DEFAULT_PROXIES = {
    "http": "socks5h://127.0.0.1:1080",
    "https": "socks5h://127.0.0.1:1080",
}


class BERTFlatClassifier(nn.Module):
    def __init__(self, model_type: str, num_classes: int = 5) -> None:
        super().__init__()
        self.model_type = model_type
        self.num_classes = num_classes
        self.encoder = AutoModel.from_pretrained(model_type)
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
        self.clf = nn.Linear(self.encoder.config.hidden_size, num_classes)

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


def check_download_resource(
    remote: RemoteFileMetadata,
    use_proxy: bool = False,
    proxies: Optional[dict[str, str]] = None,
) -> str:
    proxies = DEFAULT_PROXIES if use_proxy and proxies is None else proxies
    data_home = _get_data_home()
    file_path = os.path.join(data_home, remote.filename)
    if not os.path.exists(file_path):
        # currently don't capture error at this level, assume download success
        file_path = _download(remote, data_home, use_proxy, proxies)
    return file_path


def clear_data_home(data_home: Optional[str] = None) -> None:
    """Delete all the content of the data home cache.
    Parameters
    ----------
    data_home : str | None
        The path to data dir.
    """
    data_home = _get_data_home(data_home)
    shutil.rmtree(data_home)


def infer_preprocess(
    tokenizer: Callable, texts: list[str], max_len: int
) -> dict[str, Tensor]:
    texts = _text_preprocess(texts)  # type: ignore
    batch = tokenizer(texts, truncation=True, padding="max_length", max_length=max_len)
    for k in ["input_ids", "attention_mask", "token_type_ids"]:
        batch[k] = torch.LongTensor(batch[k])
    return batch


# - Private functions
def _text_preprocess(inp: Union[str, list[str]]) -> Union[str, list[str]]:
    if isinstance(inp, str):
        return re.sub(r"[^\w\s]", "", inp).lower()
    else:
        return [re.sub(r"[^\w\s]", "", x).lower() for x in inp]


def _get_data_home(data_home: Optional[str] = None) -> str:  # type: ignore
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


def _sha256(path: str) -> str:
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


def _download_with_bar(
    url: str, file_path: str, proxies: Optional[dict[str, str]] = DEFAULT_PROXIES
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


def _fetch_remote(
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
    file_path = _download_with_bar(remote.url, file_path, proxies)
    checksum = _sha256(file_path)
    if remote.checksum != checksum:
        raise IOError(
            "{} has an SHA256 checksum ({}) "
            "differing from expected ({}), "
            "file may be corrupted.".format(file_path, checksum, remote.checksum)
        )
    return file_path


def _download(
    remote: RemoteFileMetadata,
    file_path: Optional[str] = None,
    use_proxy: bool = False,
    proxies: Optional[dict[str, str]] = DEFAULT_PROXIES,
) -> str:
    data_home = _get_data_home()
    file_path = _fetch_remote(remote, data_home, use_proxy, proxies)
    return file_path
