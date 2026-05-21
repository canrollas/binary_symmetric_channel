"""Bit-level error metrics."""

import numpy as np
from typing import Tuple


def calculate_ber(original: np.ndarray, received: np.ndarray) -> float:
    original = np.asarray(original, dtype=np.uint8)
    received = np.asarray(received, dtype=np.uint8)
    if len(original) == 0:
        return 0.0
    return float(np.sum(original != received)) / len(original)


def calculate_ber_with_positions(original: np.ndarray, received: np.ndarray) -> Tuple[float, np.ndarray]:
    original = np.asarray(original, dtype=np.uint8)
    received = np.asarray(received, dtype=np.uint8)
    error_mask = original != received
    ber = float(np.sum(error_mask)) / len(original) if len(original) > 0 else 0.0
    return ber, np.where(error_mask)[0]
