#!/usr/bin/env python3
"""
BER (Bit Error Rate) Analysis

Plots BER graph for different error rates.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from hamming_bsc import HammingBSCSystem


def main():
    """BER analysis and plotting."""
    print("=" * 60)
    print("BER (Bit Error Rate) Analysis")
    print("=" * 60)
    
    # Test message
    message = "ARPM Project: Test Data for Deep Learning and Embedded Systems." * 10
    
    # Different error rates
    error_rates = np.linspace(0.001, 0.3, 50)
    
    ber_pre_list = []
    ber_post_list = []
    recovery_rates = []
    
    print("\nRunning simulation...")
    system = HammingBSCSystem(random_seed=42)
    
    for p in error_rates:
        system.set_error_probability(p)
        _, stats = system.process_text(message, return_stats=True)
        
        ber_pre_list.append(stats['ber_pre_correction'])
        ber_post_list.append(stats['ber_post_correction'])
        recovery_rates.append(stats['recovery_rate'])
        
        if len(ber_pre_list) % 10 == 0:
            print(f"  Progress: {len(ber_pre_list)}/{len(error_rates)}")
    
    # Plot graph
    try:
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(error_rates, ber_pre_list, 'r-', label='BER (Channel - Raw Error)', linewidth=2)
        plt.plot(error_rates, ber_post_list, 'b-', label='BER (After Correction)', linewidth=2)
        plt.plot(error_rates, error_rates, 'g--', label='Theoretical Limit (p)', linewidth=1, alpha=0.5)
        plt.xlabel('Channel Error Probability (p)', fontsize=12)
        plt.ylabel('Bit Error Rate (BER)', fontsize=12)
        plt.title('Hamming(7,4) Code Performance', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 0.3)
        plt.ylim(0, max(max(ber_pre_list), max(ber_post_list)) * 1.1)
        
        plt.subplot(2, 1, 2)
        plt.plot(error_rates, recovery_rates, 'purple', label='Recovery Rate', linewidth=2)
        plt.xlabel('Channel Error Probability (p)', fontsize=12)
        plt.ylabel('Recovery Rate', fontsize=12)
        plt.title('Error Correction Success Rate', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 0.3)
        plt.ylim(0, 1.1)
        
        plt.tight_layout()
        
        # Save graph
        output_path = project_root / "docs" / "ber_analysis.png"
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nGraph saved: {output_path}")
        
        # Show graph (if GUI available)
        try:
            plt.show()
        except:
            print("(Graph could not be displayed - headless environment)")
    
    except ImportError:
        print("\nMatplotlib not found. Graph cannot be plotted.")
        print("To install: pip install matplotlib")
        print("\nNumerical results:")
        print(f"  Lowest BER (p=0.001): {ber_post_list[0]:.6f}")
        print(f"  Medium BER (p=0.05): {ber_post_list[10]:.6f}")
        print(f"  High BER (p=0.2): {ber_post_list[-1]:.6f}")


if __name__ == "__main__":
    main()
