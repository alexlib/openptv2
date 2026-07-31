"""Coverage tests for openptv2.algorithms.lsqadj.

Pure-Python coverage only — skip this module if running against the compiled .so.
"""

import numpy as np
import pytest

# Skip entirely when running against the compiled extension.
try:
    from openptv2.algorithms.lsqadj import is_compiled as _is_compiled

    if _is_compiled():
        pytest.skip("pure-Python coverage tests only", allow_module_level=True)
except Exception:
    pass

from openptv2.algorithms.lsqadj import ata, atl, is_compiled, matinv, matmul

# ---------------------------------------------------------------------------
# is_compiled
# ---------------------------------------------------------------------------

def test_is_compiled_returns_bool():
    result = is_compiled()
    assert isinstance(result, bool)
    # In pure-Python mode cython.compiled is False
    assert result is False


# ---------------------------------------------------------------------------
# ata  — A^T @ A
# ---------------------------------------------------------------------------

class TestAta:
    def test_identity(self):
        """For identity matrix, A^T A = I."""
        a = np.eye(3)
        result = ata(a, 3, 3)
        np.testing.assert_allclose(result, np.eye(3))

    def test_simple_2x2(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = ata(a, 2, 2)
        expected = a.T @ a
        np.testing.assert_allclose(result, expected)

    def test_rectangular_m_rows_n_cols(self):
        """m=3 rows, n=2 cols — result is (2,2)."""
        a = np.array([[1.0, 2.0, 99.0],
                      [3.0, 4.0, 99.0],
                      [5.0, 6.0, 99.0]])
        result = ata(a, 3, 2)
        assert result.shape == (2, 2)
        sub = a[:, :2]
        np.testing.assert_allclose(result, sub.T @ sub)

    def test_flat_input(self):
        """Accept flat 1-D array and reshape."""
        a_2d = np.array([[1.0, 0.0], [0.0, 1.0]])
        a_flat = a_2d.ravel()
        result = ata(a_flat, 2, 2)
        np.testing.assert_allclose(result, a_2d.T @ a_2d)

    def test_single_column(self):
        a = np.array([[2.0], [3.0]])
        result = ata(a, 2, 1)
        assert result.shape == (1, 1)
        np.testing.assert_allclose(result[0, 0], 4.0 + 9.0)

    def test_dtype_coercion(self):
        """Integer input should be coerced to float64."""
        a = np.array([[1, 2], [3, 4]], dtype=np.int32)
        result = ata(a, 2, 2)
        assert result.dtype == np.float64

    def test_zero_matrix(self):
        a = np.zeros((4, 4))
        result = ata(a, 4, 4)
        np.testing.assert_allclose(result, np.zeros((4, 4)))

    def test_larger_matrix(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal((5, 5))
        result = ata(a, 5, 5)
        np.testing.assert_allclose(result, a.T @ a, atol=1e-12)

    def test_returns_ndarray(self):
        a = np.eye(2)
        result = ata(a, 2, 2)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# atl  — A^T @ l
# ---------------------------------------------------------------------------

class TestAtl:
    def test_basic(self):
        a = np.array([[1.0, 0.0], [0.0, 1.0]])
        l = np.array([3.0, 5.0])
        result = atl(a, l, 2, 2)
        np.testing.assert_allclose(result, [3.0, 5.0])

    def test_rectangular(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        l = np.array([1.0, 1.0, 1.0])
        result = atl(a, l, 3, 2)
        expected = a[:, :2].T @ l
        np.testing.assert_allclose(result, expected)

    def test_uses_n_cols_only(self):
        """Extra columns in A beyond n should be ignored."""
        a = np.array([[1.0, 2.0, 99.0], [3.0, 4.0, 99.0]])
        l = np.array([1.0, 1.0])
        result = atl(a, l, 2, 2)
        expected = a[:, :2].T @ l
        np.testing.assert_allclose(result, expected)

    def test_dtype_coercion_a(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.int32)
        l = np.array([1.0, 1.0])
        result = atl(a, l, 2, 2)
        assert result.dtype == np.float64

    def test_dtype_coercion_l(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        l = np.array([1, 2], dtype=np.int32)
        result = atl(a, l, 2, 2)
        assert result.dtype == np.float64

    def test_l_ravel(self):
        """l can be passed as a column vector."""
        a = np.array([[1.0, 0.0], [0.0, 1.0]])
        l = np.array([[2.0], [7.0]])
        result = atl(a, l, 2, 2)
        np.testing.assert_allclose(result, [2.0, 7.0])

    def test_zero_l(self):
        a = np.eye(3)
        l = np.zeros(3)
        result = atl(a, l, 3, 3)
        np.testing.assert_allclose(result, np.zeros(3))

    def test_single_column(self):
        a = np.array([[2.0], [3.0]])
        l = np.array([1.0, 1.0])
        result = atl(a, l, 2, 1)
        assert result.shape == (1,)
        np.testing.assert_allclose(result[0], 5.0)

    def test_returns_ndarray(self):
        a = np.eye(2)
        l = np.ones(2)
        result = atl(a, l, 2, 2)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# matinv  — matrix inversion
# ---------------------------------------------------------------------------

class TestMatinv:
    def test_identity(self):
        a = np.eye(3)
        result = matinv(a, 3)
        np.testing.assert_allclose(result, np.eye(3), atol=1e-12)

    def test_2x2(self):
        a = np.array([[2.0, 0.0], [0.0, 4.0]])
        result = matinv(a, 2)
        expected = np.array([[0.5, 0.0], [0.0, 0.25]])
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_general_invertible(self):
        rng = np.random.default_rng(7)
        a = rng.standard_normal((4, 4))
        # Make well-conditioned
        a = a @ a.T + 4 * np.eye(4)
        result = matinv(a, 4)
        np.testing.assert_allclose(result @ a, np.eye(4), atol=1e-10)

    def test_singular_falls_back_to_pinv(self):
        """Singular matrix should not raise; falls back to pinv."""
        a = np.zeros((3, 3))
        result = matinv(a, 3)
        assert result is not None
        assert result.shape == (3, 3)

    def test_flat_input(self):
        a = np.eye(2).ravel()
        result = matinv(a, 2)
        np.testing.assert_allclose(result, np.eye(2), atol=1e-12)

    def test_flat_input_n2(self):
        """Flat n×n array is accepted via reshape(-1, n)[:n, :n]."""
        a = np.array([2.0, 0.0, 0.0, 4.0])  # 2×2 diagonal stored flat
        result = matinv(a, 2)
        expected = np.array([[0.5, 0.0], [0.0, 0.25]])
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_dtype_coercion(self):
        a = np.eye(2, dtype=np.int32)
        result = matinv(a, 2)
        assert result.dtype == np.float64

    def test_returns_ndarray(self):
        a = np.eye(2)
        result = matinv(a, 2)
        assert isinstance(result, np.ndarray)

    def test_shape(self):
        a = np.eye(3)
        result = matinv(a, 3)
        assert result.shape == (3, 3)


# ---------------------------------------------------------------------------
# matmul  — matrix-vector product
# ---------------------------------------------------------------------------

class TestMatmul:
    def test_identity(self):
        b = np.eye(3)
        c = np.array([1.0, 2.0, 3.0])
        result = matmul(b, c, 3, 3)
        np.testing.assert_allclose(result, c)

    def test_basic(self):
        b = np.array([[1.0, 2.0], [3.0, 4.0]])
        c = np.array([1.0, 1.0])
        result = matmul(b, c, 2, 2)
        np.testing.assert_allclose(result, [3.0, 7.0])

    def test_rectangular(self):
        b = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]])
        c = np.array([5.0, 6.0])
        result = matmul(b, c, 3, 2)
        expected = b @ c
        np.testing.assert_allclose(result, expected)

    def test_subslice_rows(self):
        """m < total rows — only first m rows used."""
        b = np.array([[1.0, 0.0],
                      [0.0, 1.0],
                      [99.0, 99.0]])
        c = np.array([3.0, 4.0])
        result = matmul(b, c, 2, 2)
        assert result.shape == (2,)
        np.testing.assert_allclose(result, [3.0, 4.0])

    def test_subslice_cols(self):
        """Extra elements in c beyond n are ignored."""
        b = np.array([[1.0, 0.0], [0.0, 1.0]])
        c = np.array([7.0, 8.0, 99.0])  # only first 2 elements used
        result = matmul(b, c, 2, 2)
        np.testing.assert_allclose(result, [7.0, 8.0])

    def test_flat_b(self):
        b_2d = np.array([[2.0, 0.0], [0.0, 3.0]])
        b_flat = b_2d.ravel()
        c = np.array([1.0, 1.0])
        result = matmul(b_flat, c, 2, 2)
        np.testing.assert_allclose(result, [2.0, 3.0])

    def test_dtype_coercion(self):
        b = np.array([[1, 0], [0, 1]], dtype=np.int32)
        c = np.array([2, 3], dtype=np.int32)
        result = matmul(b, c, 2, 2)
        assert result.dtype == np.float64

    def test_zero_vector(self):
        b = np.ones((3, 3))
        c = np.zeros(3)
        result = matmul(b, c, 3, 3)
        np.testing.assert_allclose(result, np.zeros(3))

    def test_returns_ndarray(self):
        b = np.eye(2)
        c = np.ones(2)
        result = matmul(b, c, 2, 2)
        assert isinstance(result, np.ndarray)

    def test_single_element(self):
        b = np.array([[5.0]])
        c = np.array([3.0])
        result = matmul(b, c, 1, 1)
        np.testing.assert_allclose(result, [15.0])
