"""Unit tests for metrics modules."""

import numpy as np
import pytest
import sys
sys.path.insert(0, ".")

from src.metrics.shannon import binary_entropy, shannon_capacity
from src.metrics.bit_metrics import calculate_ber
from src.metrics.text_metrics import character_error_rate, word_error_rate, bleu_score, edit_distance


def test_binary_entropy_boundaries():
    assert binary_entropy(0) == 0.0
    assert binary_entropy(1) == 0.0
    assert abs(binary_entropy(0.5) - 1.0) < 1e-10


def test_shannon_capacity():
    assert abs(shannon_capacity(0) - 1.0) < 1e-10
    assert abs(shannon_capacity(0.5) - 0.0) < 1e-10
    assert 0 < shannon_capacity(0.1) < 1


def test_ber_zero():
    bits = np.array([1, 0, 1, 0], dtype=np.uint8)
    assert calculate_ber(bits, bits) == 0.0


def test_ber_all_flipped():
    bits = np.array([1, 0, 1, 0], dtype=np.uint8)
    flipped = 1 - bits
    assert calculate_ber(bits, flipped) == 1.0


def test_ber_half():
    bits = np.array([1, 0, 1, 0], dtype=np.uint8)
    recv = np.array([1, 1, 1, 0], dtype=np.uint8)
    assert abs(calculate_ber(bits, recv) - 0.25) < 1e-10


def test_edit_distance():
    assert edit_distance("", "") == 0
    assert edit_distance("abc", "abc") == 0
    assert edit_distance("abc", "abd") == 1
    assert edit_distance("abc", "") == 3


def test_cer_identical():
    assert character_error_rate("hello", "hello") == 0.0


def test_wer_identical():
    assert word_error_rate("hello world", "hello world") == 0.0


def test_bleu_identical():
    assert abs(bleu_score("hello world", "hello world") - 1.0) < 1e-6


def test_bleu_empty_hypothesis():
    assert bleu_score("hello", "") == 0.0
