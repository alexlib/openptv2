"""Matrix operations for least-squares adjustment.

Translation of lib/src/lsqadj.c and lib/include/lsqadj.h.

These are the core matrix operations used in Gauss-Markov models:
- ata: A^T @ A (normal equations matrix)
- atl: A^T @ l (normal equations RHS)
- matinv: matrix inversion via np.linalg.inv (LAPACK)
- matmul: matrix-vector multiplication
- norm_cross: normalized cross product of two 3-vectors

All functions operate on row-major (C-order) flat arrays to match
the C implementation's memory layout.
"""
import cython


import numpy as np


@cython.ccall
def ata(a: np.ndarray, m: int, n: int) -> np.ndarray:
    """Compute A^T @ A for a submatrix of A.

    Matches the C `ata()` function. Computes the product of the transpose
    of matrix A with A itself, for the first n columns.

    Args:
        a: matrix of shape (m, n_large) stored as flat array or 2D array.
        m: number of rows in A.
        n: number of columns to use (size of output matrix).

    Returns:
        ndarray of shape (n, n) = A[:, :n]^T @ A[:, :n].
    """
    a = np.asarray(a, dtype=np.float64).reshape(m, -1)
    sub = a[:, :n]
    return sub.T @ sub


@cython.ccall
def atl(a: np.ndarray, l: np.ndarray, m: int, n: int) -> np.ndarray:
    """Compute A^T @ l for a submatrix of A.

    Matches the C `atl()` function.

    Args:
        a: matrix of shape (m, n_large).
        l: vector of shape (m,).
        m: number of rows in A.
        n: number of columns to use (length of output).

    Returns:
        ndarray of shape (n,) = A[:, :n]^T @ l.
    """
    a = np.asarray(a, dtype=np.float64).reshape(m, -1)
    l = np.asarray(l, dtype=np.float64).ravel()
    sub = a[:, :n]
    return sub.T @ l


@cython.ccall
def matinv(a: np.ndarray, n: int) -> np.ndarray:
    """Invert a square matrix.

    Args:
        a: matrix of shape (n, n) or (n_large, n_large).
        n: size of the sub-matrix to invert.

    Returns:
        Inverse of a[:n, :n] as ndarray of shape (n, n).

    Raises:
        np.linalg.LinAlgError: if a is singular.
    """
    return np.linalg.inv(np.asarray(a, dtype=np.float64).reshape(-1, n)[:n, :n])


@cython.ccall
def matmul(b: np.ndarray, c: np.ndarray, m: int, n: int) -> np.ndarray:
    """Compute b @ c for submatrix b and sub-vector c.

    Matches the C `matmul()` function.

    Args:
        b: matrix of shape (m, n).
        c: vector of shape (n,).
        m: number of rows in output.
        n: number of columns.

    Returns:
        ndarray of shape (m,) = b[:m, :n] @ c[:n].
    """
    b = np.asarray(b, dtype=np.float64).reshape(-1, n)
    c = np.asarray(c, dtype=np.float64).ravel()
    return b[:m, :n] @ c[:n]


@cython.ccall
def norm_cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the normalized cross product of two 3-vectors.

    Args:
        a: first 3-vector.
        b: second 3-vector.

    Returns:
        Unit vector in direction of a x b.
    """
    from .vec_utils import vec_cross, unit_vector

    cross = vec_cross(a, b)
    return unit_vector(cross)


def is_compiled() -> bool:
    """Return whether this module is compiled to C."""
    return cython.compiled
