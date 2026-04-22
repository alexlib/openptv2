import numpy as np
import pytest
from algorithms.vec_utils import (
    vec_init, vec_set, vec_copy, vec_subt, vec_add, vec_scalar_mul, vec_norm,
    vec_diff_norm, vec_dot, vec_cross, vec_cmp, vec_approx_cmp, is_empty
)

EPS = 1e-5

def test_dot():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 2.0, 0.0])
    d = vec_dot(a, b)
    assert abs(d - 0.0) < EPS

    b2 = np.array([2.0, 2.0, 0.0])
    d2 = vec_dot(b2, a)
    assert abs(d2 - 2.0) < EPS

def test_vec_init():
    p = vec_init()
    assert all(is_empty(x) for x in p)

def test_vec_cmp():
    v1 = np.array([1., 2., 3.])
    v2 = np.array([4., 5., 6.])
    assert vec_cmp(v1, v1)
    assert not vec_cmp(v1, v2)

def test_vec_approx_cmp():
    v1 = np.array([1., 2., 3.])
    v2 = np.array([1.00001, 2.00001, 3.00001])
    assert vec_approx_cmp(v1, v2, 1e-4)
    assert not vec_approx_cmp(v1, v2, 1e-5)

def test_vec_copy():
    src = np.array([1., 2., 3.])
    dst = vec_copy(src)
    assert vec_cmp(dst, src)

def test_vec_subt():
    sub = np.array([1., 2., 3.])
    from_vec = np.array([4., 5., 6.])
    res = np.array([3., 3., 3.])
    out = vec_subt(from_vec, sub)
    assert vec_cmp(out, res)

def test_vec_add():
    vec1 = np.array([1., 2., 3.])
    vec2 = np.array([3., 3., 3.])
    res = np.array([4., 5., 6.])
    out = vec_add(vec1, vec2)
    assert vec_cmp(out, res)

def test_diff_norm():
    vec1 = np.array([1., 2., 3.])
    vec2 = np.array([4., 5., 6.])
    expected = np.sqrt(3)*3
    assert abs(vec_diff_norm(vec1, vec2) - expected) < EPS

def test_vec_set():
    res = np.array([1., 2., 3.])
    dest = vec_set(1., 2., 3.)
    assert vec_cmp(dest, res)

def test_scalar_mul():
    v1 = np.array([1., 2., 3.])
    v2 = np.array([4., 8., 12.])
    out = vec_scalar_mul(v1, 4.)
    assert vec_cmp(out, v2)

def test_unit_vec():
    v1 = np.array([1., 100., 1.])
    norm = vec_norm(v1)
    res = v1 / norm
    out = v1 / norm  # since unit_vector is just normalization
    assert vec_approx_cmp(out, res, 1e-4)

def test_cross():
    v1 = np.array([1., 0., 0.])
    v2 = np.array([0., 1., 0.])
    res = np.array([0., 0., 1.])
    out = vec_cross(v1, v2)
    assert vec_cmp(out, res)
    # parallel vectors cross = 0
    res2 = np.array([0., 0., 0.])
    out2 = vec_cross(v1, v1)
    assert vec_cmp(out2, res2)
