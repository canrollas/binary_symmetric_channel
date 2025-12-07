# Hamming Binary Symmetric Channel (BSC) System

High-performance, vectorized Hamming(7,4) encoding and Binary Symmetric Channel simulation Python library.

Optimized for efficiently processing millions of data points.

## Features

- **Hamming(7,4) Encoding**: Data encoding using systematic form
- **Binary Symmetric Channel**: Realistic noise simulation
- **Automatic Error Correction**: Automatically corrects single-bit errors
- **High Performance**: Optimized with NumPy vectorized operations
- **Batch Processing**: Efficient processing of large datasets
- **Comprehensive Tests**: Unit tests and integration tests
- **Detailed Metrics**: BER, recovery rate, and other performance metrics

## Installation

### Requirements

- Python >= 3.8
- NumPy >= 1.20.0

### Install with Pip

```bash
pip install -r requirements.txt
```

### Development Installation

```bash
pip install -e .
```

## Quick Start

### Basic Usage

```python
from hamming_bsc import HammingBSCSystem

# Create system
system = HammingBSCSystem(
    error_probability=0.05,  # 5% error rate
    random_seed=42
)

# Process text
message = "ARPM Project: Deep Learning and Embedded Systems"
result, stats = system.process_text(message, return_stats=True)

print(f"Corrected message: {result}")
print(f"BER (Channel): {stats['ber_pre_correction']:.6f}")
print(f"BER (Correction): {stats['ber_post_correction']:.6f}")
print(f"Recovery Rate: {stats['recovery_rate']:.2%}")
```

### Bit Processing

```python
import numpy as np
from hamming_bsc import HammingBSCSystem

system = HammingBSCSystem(error_probability=0.01)

# Process bit array
bits = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
recovered, stats = system.process_bits(bits, return_stats=True)

print(f"Original: {bits}")
print(f"Corrected: {recovered}")
print(f"Total corrections: {stats['corrections']}")
```

 

 