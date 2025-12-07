"""
Utility Functions for Hamming BSC System

Text-bit conversions, metric calculations, and batch processing helpers.
"""

import numpy as np
from typing import Tuple, List, Optional, Union
import multiprocessing as mp
from functools import partial


def text_to_bits(text: str, encoding: str = 'utf-8') -> np.ndarray:
    """
    Convert text to bit array.
    
    Args:
        text: Text to convert
        encoding: Character encoding (default: 'utf-8')
        
    Returns:
        Bit array (1D, dtype=uint8)
    
    Example:
        >>> bits = text_to_bits("AB")
        >>> len(bits) == 16  # 2 characters * 8 bits
        True
    """
    bits = []
    for char in text:
        # Convert to byte using UTF-8 encoding
        byte_val = char.encode(encoding)
        for byte in byte_val:
            # 8 bits for each byte
            bin_str = bin(byte)[2:].zfill(8)
            bits.extend([int(b) for b in bin_str])
    
    return np.array(bits, dtype=np.uint8)


def bits_to_text(bits: np.ndarray, encoding: str = 'utf-8') -> str:
    """
    Convert bit array to text.
    
    Args:
        bits: Bit array to convert (1D)
        encoding: Character encoding (default: 'utf-8')
        
    Returns:
        Converted text
    
    Example:
        >>> bits = text_to_bits("Test")
        >>> text = bits_to_text(bits)
        >>> text == "Test"
        True
    """
    chars = []
    byte_list = []
    
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i+8]
        if len(byte_bits) < 8:
            break
        
        # Convert 8 bits to byte
        byte_val = int("".join(map(str, byte_bits)), 2)
        byte_list.append(byte_val)
    
    # Convert byte list to string
    try:
        text = bytes(byte_list).decode(encoding, errors='ignore')
    except Exception:
        # Return empty string on decode error
        text = ""
    
    return text


def calculate_ber(original: np.ndarray, received: np.ndarray) -> float:
    """
    Bit Error Rate (BER) hesaplar.
    
    Args:
        original: Orijinal bitler
        received: Alınan/karşılaştırılacak bitler
        
    Returns:
        BER değeri (0.0 - 1.0 arası)
    
    Raises:
        ValueError: if array lengths are not equal
    
    Example:
        >>> orig = np.array([1, 0, 1, 0], dtype=np.uint8)
        >>> recv = np.array([1, 1, 1, 0], dtype=np.uint8)
        >>> calculate_ber(orig, recv)
        0.25
    """
    original = np.asarray(original, dtype=np.uint8)
    received = np.asarray(received, dtype=np.uint8)
    
    if len(original) != len(received):
        raise ValueError(f"Array lengths must be equal: {len(original)} != {len(received)}")
    
    if len(original) == 0:
        return 0.0
    
    errors = np.sum(original != received)
    return errors / len(original)


def batch_process(
    data_list: List[np.ndarray],
    process_func,
    batch_size: int = 1000,
    num_workers: Optional[int] = None,
    use_multiprocessing: bool = False
) -> List:
    """
    Process large data list in batches.
    
    Args:
        data_list: Data list to process
        process_func: Function to apply to each data item
        batch_size: Number of items per batch
        num_workers: Number of workers for parallel processing (None uses CPU count)
        use_multiprocessing: Whether to use multiprocessing
        
    Returns:
        List of processed results
    
    Example:
        >>> data = [np.array([1, 0, 1]) for _ in range(100)]
        >>> results = batch_process(data, lambda x: x * 2, batch_size=10)
        >>> len(results) == 100
        True
    """
    if not data_list:
        return []
    
    results = []
    
    if use_multiprocessing and len(data_list) > batch_size:
        # Parallel processing with multiprocessing
        if num_workers is None:
            num_workers = mp.cpu_count()
        
        # Create batches
        batches = [data_list[i:i+batch_size] for i in range(0, len(data_list), batch_size)]
        
        # Parallel processing
        with mp.Pool(num_workers) as pool:
            batch_results = pool.map(process_func, batches)
        
        # Combine results
        for batch_result in batch_results:
            if isinstance(batch_result, list):
                results.extend(batch_result)
            else:
                results.append(batch_result)
    else:
        # Sequential processing
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i+batch_size]
            batch_results = [process_func(item) for item in batch]
            results.extend(batch_results)
    
    return results


def split_into_chunks(data: np.ndarray, chunk_size: int) -> List[np.ndarray]:
    """
    Split large array into equal-sized chunks.
    
    Args:
        data: Array to split
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]


def merge_chunks(chunks: List[np.ndarray]) -> np.ndarray:
    """
    Merge chunks into a single array.
    
    Args:
        chunks: List of chunks to merge
        
    Returns:
        Merged array
    """
    if not chunks:
        return np.array([], dtype=np.uint8)
    
    return np.concatenate(chunks)

