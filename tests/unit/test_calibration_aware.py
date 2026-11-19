"""Tests for calibration_aware module."""

import numpy as np

from openptv2.calibration_aware import (
    normalize_rcm,
    multi_cycle_consistency,
    calibration_aware_threshold,
)


def test_normalize_rcm_shape():
    rcm = np.array([1.0, 2.0, 3.0, 4.0])
    pos = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
    out = normalize_rcm(rcm, pos)
    assert out.shape == rcm.shape


def test_normalize_rcm_empty():
    out = normalize_rcm(np.array([]), np.zeros((0, 3)))
    assert len(out) == 0


def test_multi_cycle_consistency_shape():
    cam = np.array([[0, 1, 2, 3], [0, 1, -1, -1], [-1, -1, -1, -1]], dtype=int)
    pos = np.zeros((3, 3))
    out = multi_cycle_consistency(cam, pos)
    assert out.shape == (3,)
    assert out[0] > out[1] > out[2]


def test_calibration_aware_threshold_shape():
    rcm = np.array([1.0, 1.0, 1.0])
    pos = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0]], dtype=float)
    out = calibration_aware_threshold(rcm, pos, base_threshold=1.0)
    assert out.shape == (3,)
    # boundary factor >= 1
    assert np.all(out >= 1.0)
    # farther points get looser threshold
    assert out[2] >= out[0]