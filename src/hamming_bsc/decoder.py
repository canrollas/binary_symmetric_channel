"""
Hamming(7,4) Decoder Implementation

Syndrome-based error detection and correction.
Automatically corrects single-bit errors.
"""

import numpy as np
from typing import Tuple, Dict


class HammingDecoder:
    """
    Hamming(7,4) Decoder
    
    Parity Check Matrix H [3x7] in [P.T | I] format
    Performs error detection and correction using syndrome calculation.
    """
    
    def __init__(self):
        """Initialize decoder and build syndrome table."""
        # Parity Check Matrix H [3x7] -> [P.T | I]
        self.H = np.array([
            [1, 1, 0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 1]
        ], dtype=np.uint8)
        
        # Syndrome -> Error Index Map (Lookup Table)
        # Columns of H matrix indicate the position of the bit pointed by syndrome
        self.syndrome_table: Dict[Tuple[int, ...], int] = {}
        self._build_syndrome_table()
        
        self._validate_matrix()
    
    def _validate_matrix(self):
        """Validate parity check matrix."""
        assert self.H.shape == (3, 7), "H matrix must be 3x7"
    
    def _build_syndrome_table(self):
        """Build syndrome table (from columns of H matrix)."""
        for idx, col in enumerate(self.H.T):
            s_val = tuple(col.tolist())
            self.syndrome_table[s_val] = idx
    
    def decode(self, received_bits: np.ndarray, pad_len: int = 0) -> Tuple[np.ndarray, int]:
        """
        Decode encoded bits and correct errors.
        
        Args:
            received_bits: Received (noisy) encoded bits (1D array)
            pad_len: Number of padding bits added during encoding
            
        Returns:
            Tuple[decoded_bits, correction_count]:
                - decoded_bits: Decoded data bits
                - correction_count: Number of errors corrected
        
        Example:
            >>> decoder = HammingDecoder()
            >>> received = np.array([1, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
            >>> decoded, corrections = decoder.decode(received)
            >>> len(decoded) == 4
            True
        """
        # Input validation
        if len(received_bits) == 0:
            return np.array([], dtype=np.uint8), 0
        
        received_bits = np.asarray(received_bits, dtype=np.uint8)
        
        # Check block count
        if len(received_bits) % 7 != 0:
            raise ValueError(f"Received bit count ({len(received_bits)}) must be multiple of 7")
        
        n_blocks = len(received_bits) // 7
        blocks = received_bits.reshape(n_blocks, 7)
        
        # Syndrome calculation: H x R.T (mod 2)
        # H (3x7), Blocks.T (7xK) -> Syndromes (3xK)
        syndromes = np.dot(self.H, blocks.T) % 2
        syndromes = syndromes.T  # Convert back to (K x 3) format
        
        corrected_blocks = blocks.copy()
        correction_count = 0
        
        # Error correction loop
        for i, syndrome in enumerate(syndromes):
            s_tuple = tuple(syndrome.tolist())
            
            # If syndrome is not 0, there is an error
            if np.any(syndrome):
                if s_tuple in self.syndrome_table:
                    # Single-bit error, correctable
                    error_idx = self.syndrome_table[s_tuple]
                    corrected_blocks[i, error_idx] = 1 - corrected_blocks[i, error_idx]
                    correction_count += 1
                # else: Double-bit error (Hamming cannot solve this, may correct wrong bit)
        
        # Extract data bits (First 4 bits of systematic code are data)
        decoded_data = corrected_blocks[:, :4].flatten()
        
        # Remove padding
        if pad_len > 0:
            decoded_data = decoded_data[:-pad_len]
        
        return decoded_data.astype(np.uint8), correction_count
    
    def decode_batch(self, received_bits_list: list, pad_list: list = None) -> Tuple[list, list]:
        """
        Decode multiple encoded blocks in batch.
        
        Args:
            received_bits_list: List of received bits
            pad_list: List of padding lengths for each block (None defaults to 0)
            
        Returns:
            Tuple[decoded_list, correction_list]:
                - decoded_list: List of decoded bits
                - correction_list: List of correction counts for each block
        """
        if pad_list is None:
            pad_list = [0] * len(received_bits_list)
        
        decoded_list = []
        correction_list = []
        
        for received_bits, pad_len in zip(received_bits_list, pad_list):
            decoded, corrections = self.decode(received_bits, pad_len)
            decoded_list.append(decoded)
            correction_list.append(corrections)
        
        return decoded_list, correction_list

