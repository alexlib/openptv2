"""Tests for ghost_components module."""

import numpy as np

from openptv2.ghost_components import (
    conflict_components,
    component_sizes,
    classify_component,
)


def test_no_points():
    pos = np.zeros((0, 3))
    cam = np.zeros((0, 2), dtype=int)
    assert conflict_components(pos, cam) == []


def test_single_point():
    pos = np.array([[1.0, 2.0, 3.0]])
    cam = np.array([[5, 7]], dtype=int)
    comps = conflict_components(pos, cam)
    assert len(comps) == 1
    assert comps[0] == [0]


def test_disjoint_targets_two_components():
    pos = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
    cam = np.array([[1, 2], [3, 4]], dtype=int)
    comps = conflict_components(pos, cam)
    assert len(comps) == 2


def test_shared_target_one_component():
    pos = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
    cam = np.array([[5, 2], [5, 9]], dtype=int)  # share target 5 in cam0
    comps = conflict_components(pos, cam)
    assert len(comps) == 1
    assert sorted(comps[0]) == [0, 1]


def test_transitive_connection():
    # 0 shares with 1, 1 shares with 2 -> one component
    pos = np.zeros((3, 3))
    cam = np.array([[1, -1], [1, 2], [-1, 2]], dtype=int)
    comps = conflict_components(pos, cam)
    assert len(comps) == 1
    assert sorted(comps[0]) == [0, 1, 2]


def test_classify():
    assert classify_component(1) == "trivial"
    assert classify_component(2) == "bipartite"
    assert classify_component(3) == "ilp"
    assert classify_component(20) == "ilp"


def test_component_sizes():
    comps = [[0], [1, 2], [3, 4, 5]]
    assert component_sizes(comps) == [1, 2, 3]