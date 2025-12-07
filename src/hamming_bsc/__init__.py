"""
Hamming Binary Symmetric Channel (BSC) Implementation

High-performance, vectorized Hamming(7,4) encoding and 
Binary Symmetric Channel simulation Python library.

Optimized for efficiently processing millions of data points.
"""

from .encoder import HammingEncoder
from .decoder import HammingDecoder
from .channel import BinarySymmetricChannel
from .system import HammingBSCSystem
from .utils import text_to_bits, bits_to_text, calculate_ber, batch_process

__version__ = "1.0.0"
__author__ = "ARPM Project Team"

__all__ = [
    "HammingEncoder",
    "HammingDecoder",
    "BinarySymmetricChannel",
    "HammingBSCSystem",
    "text_to_bits",
    "bits_to_text",
    "calculate_ber",
    "batch_process",
]

