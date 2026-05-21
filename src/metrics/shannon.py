"""Shannon information-theoretic metrics for BSC analysis."""

import numpy as np


def binary_entropy(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def shannon_capacity(p: float) -> float:
    return 1.0 - binary_entropy(p)


def capacity_curve(p_values: np.ndarray) -> np.ndarray:
    return np.array([shannon_capacity(p) for p in p_values])


def code_rate_achievable(p: float, code_rate: float) -> bool:
    return code_rate <= shannon_capacity(p)


def theoretical_hamming_ber(p: float) -> float:
    """Probability that a Hamming(7,4) block has >= 2 errors (uncorrectable)."""
    from math import comb
    return sum(comb(7, k) * (p ** k) * ((1 - p) ** (7 - k)) for k in range(2, 8))


def theoretical_repetition_ber(p: float, n: int) -> float:
    """Post-decoding BER for repetition-n code (majority vote)."""
    from math import comb
    threshold = (n + 1) // 2
    return sum(comb(n, k) * (p ** k) * ((1 - p) ** (n - k)) for k in range(threshold, n + 1))
