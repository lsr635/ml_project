# -*- coding: utf-8 -*-
import torch

from torch.utils.data import Dataset
from torch import Tensor
from typing import Optional, Callable, Any


class LangABC(Dataset):
    r"""A dummy dataset of a simple formal language.

    FORMAL LANGUAGE NOTE
    Given an alphabet :math:`\Sigma`, a formal language :math:`L` is defined as a subset of :math:`\Sigma^*`. 

    This is a NONDETERMINISTIC dummy dataset for generating strings in a formal language L,
    where :math:`L=\{w| |w| \leq \textit{max_len}, |w|_a + |w|_b = |w|_c\}, \Sigma=\{a, b, c\}`.

    Attributes:

    Args:
        size: The dummy size of the dataset, would affect the # for each epoch.
        max_len: The maximum length of strings defined for this language.
            Note: BOS and EOS not included.
        transform: The transform function for preprocessing.
    """

    def __init__(
        self, size: int, max_len: int, transform: Optional[Callable[str, Any]] = None
    ) -> None:
        super().__init__()
        self.size: int = size
        r"""
        The dummy size of the dataset,
        required to specify how many steps for each epoch.
        """
        self.max_len: int = max_len
        r"""
        The maximum length of strings defined for this language.
        Note: SOS and EOS not included.        
        """
        self.transform: Optional[Callable[str, Any]] = transform
        r"""
        The transform function for preprocessing.
        """

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Any:
        #random length from [0, max_len]
        n_c = torch.randint(self.max_len // 2 + 1, ())
        n_a = torch.randint(n_c + 1, ())
        n_b = n_c - n_a
        idx = torch.tensor([0] * n_a + [1] * n_b + [2] * n_c)
        idx = idx[torch.randperm(2*n_c)]
        out = "".join(map(lambda x: "abc"[x], idx))
        if self.transform is not None:
            out = self.transform(out)
        return out


class TokenizerABC:
    r"""A simple predefined tokenizer with 5 tokens: a, b, c, ^(BOS), and $(EOS)."""

    def __init__(self) -> None:
        self._bos:str = r"^"
        """symbol for BOS."""
        self._eos:str = r"$"
        """symbol for EOS."""
        self._tokens: list[str] = [
            self._bos,
            self._eos,
            "a",
            "b",
            "c",
        ]
        r"""All tokens in order."""
        self._token2id: dict[str, int] = {tok: id for id, tok in enumerate(self._tokens)}
        r"""Dictionary maps tokens to token ids."""
        self.num_tokens: int = len(self._tokens)
        r"""The number of tokens."""

    def encode(self, s: str, pad_to_length: Optional[int] = None) -> Tensor:
        r"""Translate a string into a Tensor of indices of tokens.

        The input string s would be padded with BOS and EOS(s).
        For example, if s is :code:`"abc"`, and pad_to_length is 10, s would be first padded as :code:`"^abc$$$$$$"`.

        Args:
            s: The input string.
            pad_to_length: If given, the input would be padded with tailling EOSs
                till its length, including BOS and EOSs, meets this value.

        Returns:
            The indices of tokens stored in an 1D tensor.
        """
        out = list(
            map(lambda tok: self._token2id[tok], [self._bos, *list(s), self._eos])
        )
        if pad_to_length is not None:
            assert (
                len(out) <= pad_to_length
            ), "Cannot pad when input is already longer than desired length!"
            out = out + [
                self._token2id[self._eos],
            ] * (pad_to_length - len(out))
        out = torch.tensor(out)
        return out

    def decode(
            self,
            idx: Tensor,
            truncate: Optional[bool] = False,
            remove_special_token: Optional[bool] = False
        ) -> str:
        r"""Translate a Tensor of indices of tokens back to string.

        Args:
            idx: The indices of tokens stored in an 1D tensor.
            truncate: If set to True, all contents after the first EOS would be removed.
            remove_special_token: BOS and EOS would be removed if set to True.

        Returns:
            The decoded string.
        """
        assert idx.dim() == 1, "The idx must be indices of a single sequence!"
        if truncate:
            eos_locations = idx.eq(self._token2id[self._eos]).nonzero(as_tuple=True)[0]
            if eos_locations.numel() > 0:
                idx = idx[:eos_locations[0]]
        if remove_special_token:
            mask = idx.not_equal(self._token2id[self._bos]) & idx.not_equal(self._token2id[self._eos])
            idx = idx[mask]
        token_list = map(lambda id: self._tokens[id], idx)
        return "".join(token_list)
