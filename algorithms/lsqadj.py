"""Matrix operations for least-squares adjustment.

Translation of lib/src/lsqadj.c and lib/include/lsqadj.h.

These are the core matrix operations used in Gauss-Markov models:
- ata: A^T @ A (normal equations matrix)
- atl: A^T @ l (normal equations RHS)
- matinv: matrix inversion via Gauss-Jordan elimination
- matmul: matrix-vector multiplication
- norm_cross: normalized cross product of two 3-vectors

All functions operate on row-major (C-order) flat arrays to match
the C implementation's memory layout.
"""

import numpy as np


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


def matinv(a: np.ndarray, n: int) -> np.ndarray:
    """Invert a square matrix via Gauss-Jordan elimination.

    Matches the C `matinv()` function which performs in-place inversion.
    This function returns a new array (does not modify input).

    Args:
        a: matrix of shape (n, n) or (n_large, n_large).
        n: size of the sub-matrix to invert.

    Returns:
        Inverse of a[:n, :n] as ndarray of shape (n, n).

    Raises:
        ZeroDivisionError: if a diagonal element is zero.
    """
    a = np.asarray(a, dtype=np.float64).reshape(-1, n)[:n, :n].copy()
    n_large = n  # in our simplified API, n_large == n

    for ipiv in range(n):
        pivot = 1.0 / a[ipiv, ipiv]
        npivot = -pivot

        # Update off-pivot elements
        for irow in range(n):
            for icol in range(n):
                if irow != ipiv and icol != ipiv:
                    a[irow, icol] -= a[ipiv, icol] * a[irow, ipiv] * pivot

        # Scale pivot row (excluding pivot element)
        for icol in range(n):
            if ipiv != icol:
                a[ipiv, icol] *= npivot

        # Scale pivot column (excluding pivot element)
        for irow in range(n):
            if ipiv != irow:
                a[irow, ipiv] *= pivot

        # Set pivot element
        a[ipiv, ipiv] = pivot

    return a


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
