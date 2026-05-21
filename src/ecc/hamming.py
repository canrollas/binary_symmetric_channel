"""Hamming(7,4) error correcting code."""

import numpy as np
from .base import ErrorCorrectingCode
from ..hamming_bsc.encoder import HammingEncoder
from ..hamming_bsc.decoder import HammingDecoder


class HammingCode(ErrorCorrectingCode):

    def __init__(self):
        self._encoder = HammingEncoder()
        self._decoder = HammingDecoder()

    @property
    def name(self) -> str:
        return "Hamming(7,4)"

    @property
    def rate(self) -> float:
        return 4 / 7

    def encode_bits(self, bits: np.ndarray) -> np.ndarray:
        encoded, self._pad_len = self._encoder.encode(bits)
        return encoded

    def decode_bits(self, bits: np.ndarray) -> np.ndarray:
        pad_len = getattr(self, "_pad_len", 0)
        decoded, _ = self._decoder.decode(bits, pad_len)
        return decoded
