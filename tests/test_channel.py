"""Tests for BSC channel."""

import numpy as np
import pytest
import sys
sys.path.insert(0, ".")

from src.channel.bsc import BSC


def test_zero_p_no_flips():
    ch = BSC(0.0, seed=0)
    bits = np.ones(1000, dtype=np.uint8)
    noisy, _ = ch.transmit(bits)
    assert np.all(bits == noisy)


def test_one_p_all_flipped():
    ch = BSC(1.0, seed=0)
    bits = np.ones(1000, dtype=np.uint8)
    noisy, _ = ch.transmit(bits)
    assert np.all(noisy == 0)


def test_empirical_ber_converges():
    p = 0.15
    ch = BSC(p, seed=42)
    bits = np.zeros(200_000, dtype=np.uint8)
    noisy, n_errors = ch.transmit(bits)
    empirical = n_errors / len(bits)
    assert abs(empirical - p) < 0.005


def test_transmit_with_positions():
    ch = BSC(0.1, seed=7)
    bits = np.zeros(500, dtype=np.uint8)
    noisy, flips = ch.transmit_with_positions(bits)
    assert np.all(noisy[flips] == 1)
    assert np.all(noisy[np.setdiff1d(np.arange(500), flips)] == 0)


def test_capacity_range():
    assert BSC(0.0).capacity() == 1.0
    assert abs(BSC(0.5).capacity()) < 1e-10
    assert 0 < BSC(0.1).capacity() < 1
