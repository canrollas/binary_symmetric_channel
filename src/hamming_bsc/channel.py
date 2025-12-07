"""
Binary Symmetric Channel (BSC) Implementation

Noise channel model where each bit is flipped with probability 'p'.
High-performance, vectorized implementation.
"""

import numpy as np
from typing import Tuple, Optional


class BinarySymmetricChannel:
    """
    Binary Symmetric Channel (BSC)
    
    Each bit is flipped independently with probability 'p'.
    Noise model: P(0->1) = P(1->0) = p
    """
    
    def __init__(self, error_probability: float, random_seed: Optional[int] = None):
        """
        Initialize BSC channel.
        
        Args:
            error_probability: Bit flip probability (0 <= p <= 1)
            random_seed: Seed for random number generator (for reproducibility)
        
        Raises:
            ValueError: if error_probability is not in [0, 1] range
        """
        if not 0 <= error_probability <= 1:
            raise ValueError(f"Error probability must be in [0, 1] range, got {error_probability}")
        
        self.p = error_probability
        self.rng = np.random.default_rng(random_seed)
    
    def transmit(self, bits: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Transmit bits through channel (adds noise).
        
        Args:
            bits: Bits to transmit (1D array)
            
        Returns:
            Tuple[noisy_bits, num_errors]:
                - noisy_bits: Noisy (flipped) bits
                - num_errors: Total number of flipped bits
        
        Example:
            >>> channel = BinarySymmetricChannel(0.1, random_seed=42)
            >>> bits = np.array([1, 0, 1, 0, 1], dtype=np.uint8)
            >>> noisy, errors = channel.transmit(bits)
            >>> len(noisy) == len(bits)
            True
        """
        if len(bits) == 0:
            return np.array([], dtype=np.uint8), 0
        
        bits = np.asarray(bits, dtype=np.uint8)
        
        # Noise mask: boolean array that is True with probability p
        noise_mask = self.rng.random(len(bits)) < self.p
        
        # Flip operation using XOR (vectorized)
        noisy_bits = bits ^ noise_mask.astype(np.uint8)
        
        # Statistics: count how many bits were flipped
        num_errors = np.sum(noise_mask)
        
        return noisy_bits, num_errors
    
    def transmit_batch(self, bits_list: list) -> Tuple[list, list]:
        """
        Transmit multiple bit blocks through channel in batch.
        
        Args:
            bits_list: List of bits to transmit
            
        Returns:
            Tuple[noisy_list, error_list]:
                - noisy_list: List of noisy bits
                - error_list: List of error counts for each block
        """
        noisy_list = []
        error_list = []
        
        for bits in bits_list:
            noisy, errors = self.transmit(bits)
            noisy_list.append(noisy)
            error_list.append(errors)
        
        return noisy_list, error_list
    
    def set_error_probability(self, error_probability: float):
        """
        Update error probability.
        
        Args:
            error_probability: New bit flip probability (0 <= p <= 1)
        """
        if not 0 <= error_probability <= 1:
            raise ValueError(f"Error probability must be in [0, 1] range, got {error_probability}")
        self.p = error_probability
    
    def get_error_probability(self) -> float:
        """Return current error probability."""
        return self.p

