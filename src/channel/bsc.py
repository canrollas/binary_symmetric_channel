"""Extended Binary Symmetric Channel with research utilities."""

import numpy as np
from typing import Tuple, Optional
from ..hamming_bsc.channel import BinarySymmetricChannel
from ..metrics.shannon import shannon_capacity, theoretical_hamming_ber


class BSC:
    """Research-grade wrapper around BinarySymmetricChannel."""

    def __init__(self, p: float, seed: Optional[int] = None):
        if not 0 <= p <= 1:
            raise ValueError(f"p must be in [0,1], got {p}")
        self.p = p
        self._ch = BinarySymmetricChannel(p, random_seed=seed)

    def transmit(self, bits: np.ndarray) -> Tuple[np.ndarray, int]:
        return self._ch.transmit(bits)

    def transmit_with_positions(self, bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        noisy, _ = self._ch.transmit(bits)
        flip_indices = np.where(bits != noisy)[0]
        return noisy, flip_indices

    def reset(self, seed: int):
        self._ch.rng = np.random.default_rng(seed)

    def set_p(self, p: float):
        self.p = p
        self._ch.set_error_probability(p)

    def capacity(self) -> float:
        return shannon_capacity(self.p)

    def theoretical_hamming_ber(self) -> float:
        return theoretical_hamming_ber(self.p)
