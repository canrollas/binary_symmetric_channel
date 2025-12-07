#!/usr/bin/env python3
"""
Batch Processing Example

Example of efficiently processing millions of data points.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from hamming_bsc import HammingEncoder, HammingDecoder, BinarySymmetricChannel
from hamming_bsc.utils import batch_process, split_into_chunks


def process_single_block(data):
    """Process a single data block."""
    encoder = HammingEncoder()
    decoder = HammingDecoder()
    channel = BinarySymmetricChannel(error_probability=0.05, random_seed=42)
    
    # Encode
    encoded, pad = encoder.encode(data)
    
    # Channel
    noisy, _ = channel.transmit(encoded)
    
    # Decode
    recovered, _ = decoder.decode(noisy, pad)
    
    return recovered


def main():
    """Batch processing example."""
    print("=" * 60)
    print("Batch Processing - Large Dataset Processing")
    print("=" * 60)
    
    # Create large dataset (1 million bits)
    print("\n1. Creating dataset...")
    total_bits = 1_000_000
    data = np.random.randint(0, 2, size=total_bits, dtype=np.uint8)
    print(f"   Total bits: {total_bits:,}")
    
    # Split into chunks (1000 bits per chunk)
    chunk_size = 1000
    chunks = split_into_chunks(data, chunk_size)
    print(f"   Number of chunks: {len(chunks):,} (each {chunk_size} bits)")
    
    # Process with batch processing
    print("\n2. Starting batch processing...")
    start_time = time.time()
    
    results = batch_process(
        chunks,
        process_single_block,
        batch_size=100,
        use_multiprocessing=False  # Set to True to test parallel processing
    )
    
    elapsed_time = time.time() - start_time
    
    # Merge results
    recovered = np.concatenate(results)
    
    # Performance metrics
    print(f"\n3. Processing completed!")
    print(f"   Time: {elapsed_time:.2f} seconds")
    print(f"   Speed: {total_bits / elapsed_time:,.0f} bits/second")
    print(f"   Speed: {len(chunks) / elapsed_time:,.0f} chunks/second")
    
    # Accuracy check
    errors = np.sum(data != recovered)
    ber = errors / len(data)
    print(f"\n4. Accuracy:")
    print(f"   Total errors: {errors:,}")
    print(f"   BER: {ber:.6f}")
    print(f"   Success rate: {(1 - ber):.2%}")
    
    # Performance comparison with different batch sizes
    print("\n" + "=" * 60)
    print("Performance Comparison with Different Batch Sizes")
    print("=" * 60)
    
    batch_sizes = [10, 50, 100, 500, 1000]
    
    for bs in batch_sizes:
        start = time.time()
        _ = batch_process(chunks, process_single_block, batch_size=bs, use_multiprocessing=False)
        elapsed = time.time() - start
        print(f"Batch Size {bs:4d}: {elapsed:.2f} seconds ({total_bits/elapsed:,.0f} bits/s)")


if __name__ == "__main__":
    main()
