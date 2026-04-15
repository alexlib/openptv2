"""Unit tests for vec_utils module.

Each function is tested with explicit known inputs and expected outputs,
following the pattern of lib/tests/check_vec_utils.c.
"""

import math

import numpy as np
import pytest

from algorithms.vec_utils import (
    norm,
    unit_vector,
    vec_add,
    vec_cross,
    vec_diff_norm,
    vec_dot,
    vec_norm,
    vec_scalar_mul,
    vec_set,
    vec_subt,
)

EPS = 1e-5


def test_vec_dot_orthogonal():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 2.0, 0.0])
    assert abs(vec_dot(a, b) - 0.0) < EPS


def test_vec_dot_nonzero():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([2.0, 2.0, 0.0])
    assert abs(vec_dot(b, a) - 2.0) < EPS


def test_vec_norm_pythagorean():
    v = np.array([3.0, 4.0, 0.0])
    assert abs(vec_norm(v) - 5.0) < EPS


def test_norm_pythagorean():
    assert abs(norm(3.0, 4.0, 0.0) - 5.0) < EPS


def test_vec_set_values():
    v = vec_set(1.0, 2.0, 3.0)
    assert abs(v[0] - 1.0) < EPS
    assert abs(v[1] - 2.0) < EPS
    assert abs(v[2] - 3.0) < EPS


def test_vec_subt():
    from_ = np.array([4.0, 5.0, 6.0])
    sub = np.array([1.0, 2.0, 3.0])
    expected = np.array([3.0, 3.0, 3.0])
    np.testing.assert_allclose(vec_subt(from_, sub), expected, atol=EPS)


def test_vec_add():
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([3.0, 3.0, 3.0])
    expected = np.array([4.0, 5.0, 6.0])
    np.testing.assert_allclose(vec_add(v1, v2), expected, atol=EPS)


def test_vec_scalar_mul():
    v = np.array([1.0, 2.0, 3.0])
    expected = np.array([4.0, 8.0, 12.0])
    np.testing.assert_allclose(vec_scalar_mul(v, 4.0), expected, atol=EPS)


def test_unit_vector():
    v = np.array([1.0, 100.0, 1.0])
    u = unit_vector(v)
    expected = np.array([0.0099, 0.9999, 0.0099])
    np.testing.assert_allclose(u, expected, atol=1e-4)


def test_unit_vector_is_normalized():
    v = np.array([3.0, 4.0, 0.0])
    u = unit_vector(v)
    assert abs(np.linalg.norm(u) - 1.0) < EPS


def test_vec_diff_norm():
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([4.0, 5.0, 6.0])
    expected = math.sqrt(3) * 3.0
    assert abs(vec_diff_norm(v1, v2) - expected) < EPS


def test_vec_cross_perpendicular():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    expected = np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(vec_cross(v1, v2), expected, atol=EPS)


def test_vec_cross_parallel_is_zero():
    v1 = np.array([1.0, 0.0, 0.0])
    result = vec_cross(v1, v1)
    np.testing.assert_allclose(result, np.zeros(3), atol=EPS)
