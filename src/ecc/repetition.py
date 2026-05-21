"""Repetition code with majority-vote decoding."""

import numpy as np
from .base import ErrorCorrectingCode


class RepetitionCode(ErrorCorrectingCode):

    def __init__(self, n: int = 3):
        if n % 2 == 0:
            raise ValueError("n must be odd for unambiguous majority vote")
        self.n = n

    @property
    def name(self) -> str:
        return f"Repetition({self.n})"

    @property
    def rate(self) -> float:
        return 1 / self.n

    def encode_bits(self, bits: np.ndarray) -> np.ndarray:
        return np.repeat(bits, self.n)

    def decode_bits(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.uint8)
        # Pad to multiple of n
        remainder = len(bits) % self.n
        if remainder:
            bits = np.pad(bits, (0, self.n - remainder))
        groups = bits.reshape(-1, self.n)
        # Majority vote
        return (groups.sum(axis=1) > self.n / 2).astype(np.uint8)
