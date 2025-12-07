"""
Hamming(7,4) Encoder Implementation

Encodes data bits using systematic form.
Produces 7 encoded bits for every 4 data bits.
"""

import numpy as np
from typing import Tuple


class HammingEncoder:
    """
    Hamming(7,4) Encoder
    
    Generator Matrix G [4x7] in [I | P] format
    First 4 bits are data, last 3 bits are parity bits.
    """
    
    def __init__(self):
        """Initialize encoder and define matrices."""
        # Generator Matrix G [4x7] -> [I | P]
        # Systematic form: First 4 bits are data, last 3 bits are parity
        self.G = np.array([
            [1, 0, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1]
        ], dtype=np.uint8)
        
        self._validate_matrix()
    
    def _validate_matrix(self):
        """Validate generator matrix."""
        assert self.G.shape == (4, 7), "G matrix must be 4x7"
        assert np.array_equal(self.G[:, :4], np.eye(4, dtype=np.uint8)), \
            "G matrix must be in systematic form (first 4 columns are identity)"
    
    def encode(self, data_bits: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Encode data bits using Hamming(7,4) code.
        
        Args:
            data_bits: Data bits to encode (1D array, dtype=uint8 or bool)
            
        Returns:
            Tuple[encoded_bits, pad_length]:
                - encoded_bits: Encoded bits (1D array)
                - pad_length: Number of padding bits added (0-3)
        
        Example:
            >>> encoder = HammingEncoder()
            >>> data = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
            >>> encoded, pad = encoder.encode(data)
            >>> len(encoded) == 14  # 8 bit -> 2 blok -> 14 bit
            True
        """
        # Input validation
        if len(data_bits) == 0:
            return np.array([], dtype=np.uint8), 0
        
        data_bits = np.asarray(data_bits, dtype=np.uint8)
        
        # Padding: Length must be multiple of 4
        pad_len = (4 - len(data_bits) % 4) % 4
        padded_data = np.pad(data_bits, (0, pad_len), 'constant', constant_values=0)
        
        # Reshape: Organize into (K x 4) blocks
        n_blocks = len(padded_data) // 4
        blocks = padded_data.reshape(n_blocks, 4)
        
        # Matrix multiplication: Blocks x G (mod 2)
        # (K x 4) dot (4 x 7) -> (K x 7)
        encoded_blocks = np.dot(blocks, self.G) % 2
        
        # Flatten and convert to uint8
        encoded_bits = encoded_blocks.flatten().astype(np.uint8)
        
        return encoded_bits, pad_len
    
    def encode_batch(self, data_bits_list: list) -> Tuple[list, list]:
        """
        Encode multiple data blocks in batch.
        
        Args:
            data_bits_list: List of data bits to encode
            
        Returns:
            Tuple[encoded_list, pad_list]:
                - encoded_list: List of encoded bits
                - pad_list: List of padding lengths for each block
        """
        encoded_list = []
        pad_list = []
        
        for data_bits in data_bits_list:
            encoded, pad = self.encode(data_bits)
            encoded_list.append(encoded)
            pad_list.append(pad)
        
        return encoded_list, pad_list

