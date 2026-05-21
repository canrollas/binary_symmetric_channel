"""Tests for ECC implementations."""

import numpy as np
import pytest
import sys
sys.path.insert(0, ".")

from src.ecc.hamming import HammingCode
from src.ecc.repetition import RepetitionCode
from src.channel.bsc import BSC


# ---- Hamming ----

def test_hamming_rate():
    assert abs(HammingCode().rate - 4 / 7) < 1e-9


def test_hamming_zero_noise_roundtrip():
    code = HammingCode()
    ch = BSC(0.0, seed=0)
    result = code.process("Hello, world!", ch)
    assert result["original"] == result["recovered"]
    assert result["ber_post"] == 0.0


def test_hamming_corrects_single_bit_errors():
    code = HammingCode()
    text = "Test message for error correction."
    from src.hamming_bsc.utils import text_to_bits
    bits = text_to_bits(text)
    encoded = code.encode_bits(bits)
    # Flip exactly one bit per 7-bit block
    corrupted = encoded.copy()
    for i in range(0, len(corrupted) - 6, 7):
        corrupted[i] ^= 1
    recovered_bits = code.decode_bits(corrupted)
    from src.hamming_bsc.utils import bits_to_text
    recovered = bits_to_text(recovered_bits)
    assert recovered == text


# ---- Repetition ----

def test_repetition_rate():
    assert RepetitionCode(3).rate == pytest.approx(1 / 3)
    assert RepetitionCode(5).rate == pytest.approx(1 / 5)


def test_repetition_odd_only():
    with pytest.raises(ValueError):
        RepetitionCode(4)


def test_repetition_corrects_minority_flips():
    code = RepetitionCode(3)
    bits = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
    encoded = code.encode_bits(bits)
    # Flip second bit in each triple (minority)
    corrupted = encoded.copy()
    corrupted[1::3] ^= 1
    recovered = code.decode_bits(corrupted)
    assert np.array_equal(recovered, bits)


def test_repetition_zero_noise_roundtrip():
    code = RepetitionCode(3)
    ch = BSC(0.0, seed=0)
    result = code.process("Shannon limit experiment", ch)
    assert result["original"] == result["recovered"]


def test_repetition5_theoretical_ber():
    """Empirical BER matches theoretical formula within tolerance."""
    from src.metrics.shannon import theoretical_repetition_ber
    p = 0.10
    n_bits = 100_000
    code = RepetitionCode(5)
    ch = BSC(p, seed=123)
    bits = np.random.default_rng(0).integers(0, 2, n_bits, dtype=np.uint8)
    encoded = code.encode_bits(bits)
    noisy, _ = ch.transmit(encoded)
    recovered = code.decode_bits(noisy)
    min_len = min(len(bits), len(recovered))
    empirical = np.sum(bits[:min_len] != recovered[:min_len]) / min_len
    theoretical = theoretical_repetition_ber(p, 5)
    assert abs(empirical - theoretical) < 0.01
