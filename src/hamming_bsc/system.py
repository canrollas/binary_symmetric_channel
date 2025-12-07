"""
Hamming BSC System - Complete Pipeline

Complete system implementation combining Encoder, Channel, and Decoder.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, Union
from .encoder import HammingEncoder
from .decoder import HammingDecoder
from .channel import BinarySymmetricChannel
from .utils import text_to_bits, bits_to_text


class HammingBSCSystem:
    """
    Hamming(7,4) + Binary Symmetric Channel Complete System
    
    Source -> Encoder -> Channel -> Decoder -> Output pipeline
    """
    
    def __init__(
        self,
        error_probability: float = 0.05,
        random_seed: Optional[int] = None
    ):
        """
        Initialize system.
        
        Args:
            error_probability: Bit flip probability for BSC channel (0 <= p <= 1)
            random_seed: Seed for random number generator (reproducibility)
        """
        self.encoder = HammingEncoder()
        self.decoder = HammingDecoder()
        self.channel = BinarySymmetricChannel(error_probability, random_seed)
    
    def process_text(
        self,
        text: str,
        return_stats: bool = False
    ) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """
        Process text through complete pipeline.
        
        Args:
            text: Text to process
            return_stats: Whether to return statistics
            
        Returns:
            If return_stats=False: Corrected text
            If return_stats=True: (corrected_text, statistics) tuple
        
        Example:
            >>> system = HammingBSCSystem(error_probability=0.01, random_seed=42)
            >>> result = system.process_text("Test")
            >>> isinstance(result, str)
            True
        """
        # 1. Convert text to bits
        original_bits = text_to_bits(text)
        
        # 2. Encode
        encoded_bits, pad_len = self.encoder.encode(original_bits)
        
        # 3. Channel (BSC)
        noisy_bits, channel_errors = self.channel.transmit(encoded_bits)
        
        # 4. Decode
        recovered_bits, corrections = self.decoder.decode(noisy_bits, pad_len)
        
        # 5. Convert bits to text
        recovered_text = bits_to_text(recovered_bits)
        
        if not return_stats:
            return recovered_text
        
        # Calculate statistics
        final_errors = np.sum(original_bits != recovered_bits)
        ber_pre = channel_errors / len(encoded_bits) if len(encoded_bits) > 0 else 0.0
        ber_post = final_errors / len(original_bits) if len(original_bits) > 0 else 0.0
        
        stats = {
            'original_length': len(text),
            'original_bits': len(original_bits),
            'encoded_bits': len(encoded_bits),
            'channel_errors': int(channel_errors),
            'corrections': int(corrections),
            'final_errors': int(final_errors),
            'ber_pre_correction': float(ber_pre),
            'ber_post_correction': float(ber_post),
            'recovery_rate': float(1.0 - ber_post) if ber_pre > 0 else 1.0
        }
        
        return recovered_text, stats
    
    def process_bits(
        self,
        bits: np.ndarray,
        return_stats: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Process bit array through complete pipeline.
        
        Args:
            bits: Bit array to process
            return_stats: Whether to return statistics
            
        Returns:
            If return_stats=False: Corrected bits
            If return_stats=True: (corrected_bits, statistics) tuple
        """
        bits = np.asarray(bits, dtype=np.uint8)
        
        # 1. Encode
        encoded_bits, pad_len = self.encoder.encode(bits)
        
        # 2. Channel (BSC)
        noisy_bits, channel_errors = self.channel.transmit(encoded_bits)
        
        # 3. Decode
        recovered_bits, corrections = self.decoder.decode(noisy_bits, pad_len)
        
        if not return_stats:
            return recovered_bits
        
        # Calculate statistics
        final_errors = np.sum(bits != recovered_bits)
        ber_pre = channel_errors / len(encoded_bits) if len(encoded_bits) > 0 else 0.0
        ber_post = final_errors / len(bits) if len(bits) > 0 else 0.0
        
        stats = {
            'original_bits': len(bits),
            'encoded_bits': len(encoded_bits),
            'channel_errors': int(channel_errors),
            'corrections': int(corrections),
            'final_errors': int(final_errors),
            'ber_pre_correction': float(ber_pre),
            'ber_post_correction': float(ber_post),
            'recovery_rate': float(1.0 - ber_post) if ber_pre > 0 else 1.0
        }
        
        return recovered_bits, stats
    
    def set_error_probability(self, error_probability: float):
        """Update BSC channel error probability."""
        self.channel.set_error_probability(error_probability)
    
    def get_error_probability(self) -> float:
        """Return current error probability."""
        return self.channel.get_error_probability()

