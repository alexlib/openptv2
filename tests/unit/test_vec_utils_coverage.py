import numpy as np
import pytest

from openptv2.algorithms.vec_utils import (
    Vec3dBatch,
    is_compiled,
    is_empty,
    unit_vector,
    vec_add,
    vec_approx_cmp,
    vec_cmp,
    vec_copy,
    vec_cross,
    vec_diff_norm,
    vec_dot,
    vec_init,
    vec_norm,
    vec_scalar_mul,
    vec_set,
    vec_subt,
)

EPS = 1e-10


def test_is_empty():
    assert is_empty(np.nan)
    assert not is_empty(0.0)


def test_vec_init_all_nan():
    assert np.all(np.isnan(vec_init()))


def test_vec_set_components():
    assert np.array_equal(vec_set(1.0, 2.0, 3.0), [1.0, 2.0, 3.0])


def test_vec_copy_independent():
    src = np.array([1.0, 2.0, 3.0])
    dst = vec_copy(src)
    assert np.array_equal(dst, src)
    dst[0] = 99.0
    assert src[0] == 1.0


def test_vec_subt_add_inverse():
    a = np.array([4.0, 5.0, 6.0])
    b = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(vec_add(vec_subt(a, b), b), a)


def test_vec_scalar_mul():
    assert np.array_equal(vec_scalar_mul(np.array([1.0, 2.0, 3.0]), 2.0), [2.0, 4.0, 6.0])


def test_vec_diff_norm():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([3.0, 4.0, 0.0])
    assert abs(vec_diff_norm(a, b) - 5.0) < EPS


def test_vec_dot():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    assert abs(vec_dot(a, b) - 32.0) < EPS


def test_vec_cross_orthogonal():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert np.array_equal(vec_cross(a, b), [0.0, 0.0, 1.0])


def test_vec_cmp():
    a = np.array([1.0, 2.0, 3.0])
    assert vec_cmp(a, a.copy())
    assert not vec_cmp(a, np.array([1.0, 2.0, 4.0]))


def test_vec_approx_cmp():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0 + 1e-12, 2.0, 3.0])
    assert vec_approx_cmp(a, b)
    assert not vec_approx_cmp(a, np.array([1.5, 2.0, 3.0]))


def test_unit_vector_normalizes():
    v = np.array([3.0, 0.0, 4.0])
    out = unit_vector(v)
    assert abs(vec_norm(out) - 1.0) < EPS
    # direction preserved
    assert np.allclose(out, v / 5.0)


def test_unit_vector_zero_unchanged():
    v = np.array([0.0, 0.0, 0.0])
    out = unit_vector(v)
    assert np.array_equal(out, v)


def test_is_compiled_returns_bool():
    assert isinstance(is_compiled(), bool)


def test_batch_empty_init():
    b = Vec3dBatch()
    assert len(b) == 0
    assert b.x.shape == (0,)


def test_batch_init_from_components():
    b = Vec3dBatch([1.0, 2.0], [3.0, 4.0], [5.0, 6.0])
    assert len(b) == 2
    assert np.array_equal(b.x, [1.0, 2.0])


def test_batch_init_shape_mismatch_raises():
    with pytest.raises(ValueError):
        Vec3dBatch([1.0, 2.0], [3.0], [5.0, 6.0])


def test_batch_from_array_roundtrip():
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = Vec3dBatch.from_array(arr)
    assert np.array_equal(b.to_array(), arr)


def test_batch_from_array_bad_shape_raises():
    with pytest.raises(ValueError):
        Vec3dBatch.from_array(np.zeros((2, 4)))


def test_batch_to_array_empty():
    b = Vec3dBatch()
    assert b.to_array().shape == (0, 3)


def test_batch_getitem_setitem():
    b = Vec3dBatch([1.0], [2.0], [3.0])
    assert np.array_equal(b[0], [1.0, 2.0, 3.0])
    b[0] = np.array([7.0, 8.0, 9.0])
    assert np.array_equal(b[0], [7.0, 8.0, 9.0])


def test_batch_add_subtract_inverse():
    a = Vec3dBatch([1.0, 2.0], [3.0, 4.0], [5.0, 6.0])
    c = Vec3dBatch([0.5, 0.5], [0.5, 0.5], [0.5, 0.5])
    back = a.add(c).subtract(c)
    assert np.allclose(back.to_array(), a.to_array())


def test_batch_scalar_mul():
    a = Vec3dBatch([1.0], [2.0], [3.0])
    out = a.scalar_mul(2.0)
    assert np.array_equal(out.to_array(), [[2.0, 4.0, 6.0]])


def test_batch_norms():
    a = Vec3dBatch([3.0], [0.0], [4.0])
    assert np.allclose(a.norms(), [5.0])


def test_batch_diff_norms():
    a = Vec3dBatch([0.0], [0.0], [0.0])
    b = Vec3dBatch([3.0], [4.0], [0.0])
    assert np.allclose(a.diff_norms(b), [5.0])


def test_batch_dot():
    a = Vec3dBatch([1.0], [2.0], [3.0])
    b = Vec3dBatch([4.0], [5.0], [6.0])
    assert np.allclose(a.dot(b), [32.0])


def test_batch_cross_orthogonal():
    a = Vec3dBatch([1.0], [0.0], [0.0])
    b = Vec3dBatch([0.0], [1.0], [0.0])
    assert np.allclose(a.cross(b).to_array(), [[0.0, 0.0, 1.0]])


def test_batch_unit_norms_are_one():
    a = Vec3dBatch([3.0, 0.0], [0.0, 0.0], [4.0, 0.0])
    u = a.unit()
    norms = u.norms()
    # first vector normalized to 1, zero vector stays zero
    assert abs(norms[0] - 1.0) < EPS
    assert abs(norms[1] - 0.0) < EPS


def test_batch_init_empty_all_nan():
    b = Vec3dBatch().init_empty(3)
    assert len(b) == 3
    assert np.all(np.isnan(b.to_array()))
