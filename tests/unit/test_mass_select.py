"""Capacity-aware selection from the candidate pool (scripts/joint_bounds.py::mass_select)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import joint_bounds as jb  # noqa: E402

pytestmark = pytest.mark.unit


def test_a_bright_blob_serves_two_points_and_sub_triplets_are_not_taken_twice():
    # camera 0 blob 0 holds two particles (capacity 2); particles A = (0,0,0,0) and B = (0,1,1,1)
    ids = np.array([[0, 0, 0, 0],      # A quad
                    [0, 1, 1, 1],      # B quad, shares camera-0 blob 0 with A
                    [0, 0, 0, -1],     # sub-triplet of A: shares 3 blobs with A
                    [0, 0, 1, -1]])    # ghost mixing A and B
    corr = np.array([10.0, 9.0, 8.0, 7.0])
    cap = [np.array([2, 5]), np.array([1, 1]), np.array([1, 1]), np.array([1, 1])]
    sel = jb.mass_select(ids, corr, cap)
    assert sorted(sel.tolist()) == [0, 1]


def test_capacity_one_behaves_like_the_unique_matcher():
    ids = np.array([[0, 0, 0, 0], [0, 1, 1, 1]])
    cap = [np.array([1, 1])] * 4
    assert jb.mass_select(ids, np.array([2.0, 1.0]), cap).tolist() == [0]
