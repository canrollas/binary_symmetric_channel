#!/usr/bin/env python3
"""
Basic Usage Example

Demonstrates simple usage of Hamming BSC system.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from hamming_bsc import HammingBSCSystem


def main():
    """Basic usage example."""
    print("=" * 60)
    print("Hamming BSC System - Basic Usage Example")
    print("=" * 60)
    
    # Create system
    system = HammingBSCSystem(
        error_probability=0.05,  # 5% error rate
        random_seed=42  # For reproducibility
    )
    
    # Test message
    message = "ARPM Project: Test Data for Deep Learning and Embedded Systems."
    
    print(f"\nOriginal Message: {message}")
    print(f"Message Length: {len(message)} characters")
    
    # Process through pipeline
    recovered_text, stats = system.process_text(message, return_stats=True)
    
    # Show results
    print("\n" + "-" * 60)
    print("STATISTICS")
    print("-" * 60)
    print(f"Original Bit Count: {stats['original_bits']}")
    print(f"Encoded Bit Count: {stats['encoded_bits']}")
    print(f"Channel Errors: {stats['channel_errors']}")
    print(f"Corrections: {stats['corrections']}")
    print(f"Remaining Errors: {stats['final_errors']}")
    print(f"BER (Channel): {stats['ber_pre_correction']:.6f}")
    print(f"BER (After Correction): {stats['ber_post_correction']:.6f}")
    print(f"Recovery Rate: {stats['recovery_rate']:.2%}")
    print("-" * 60)
    
    print(f"\nCorrected Message: {recovered_text}")
    print(f"\nMessages Equal? {message == recovered_text}")
    
    # Test with different error rates
    print("\n" + "=" * 60)
    print("Testing with Different Error Rates")
    print("=" * 60)
    
    test_rates = [0.001, 0.01, 0.05, 0.1, 0.2]
    
    for p in test_rates:
        system.set_error_probability(p)
        _, stats = system.process_text(message, return_stats=True)
        print(f"\np = {p:.3f}:")
        print(f"  BER (Channel): {stats['ber_pre_correction']:.6f}")
        print(f"  BER (Correction): {stats['ber_post_correction']:.6f}")
        print(f"  Recovery: {stats['recovery_rate']:.2%}")


if __name__ == "__main__":
    main()
