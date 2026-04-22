"""Parity helper utilities for Python vs optv comparisons."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def optv_available() -> bool:
    """Return True if optv bindings are importable in this environment."""
    try:
        import optv  # noqa: F401

        return True
    except Exception:
        return False


def assert_array_allclose(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    rtol: float,
    atol: float,
    msg: str = "",
) -> None:
    """Assert shape, dtype-kind compatibility and numeric closeness."""
    assert actual.shape == expected.shape, f"shape mismatch: {actual.shape} != {expected.shape}. {msg}"
    assert actual.dtype.kind == expected.dtype.kind, (
        f"dtype kind mismatch: {actual.dtype} != {expected.dtype}. {msg}"
    )
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def sorted_tuple_rows(arr: np.ndarray) -> np.ndarray:
    """Return rows sorted lexicographically for stable comparison."""
    if arr.size == 0:
        return arr
    keys = tuple(arr[:, i] for i in range(arr.shape[1] - 1, -1, -1))
    order = np.lexsort(keys)
    return arr[order]


def unpack_correspondence_result(res: np.recarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract tuple indices and correlation arrays from correspondence output."""
    if len(res) == 0:
        return np.empty((0, 4), dtype=np.int32), np.empty((0,), dtype=np.float64)
    return np.asarray(res.p, dtype=np.int32), np.asarray(res.corr, dtype=np.float64)
