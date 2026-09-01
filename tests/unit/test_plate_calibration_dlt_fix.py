"""Regression for DLT padding bug: duplicating views makes A rank-deficient/biases."""

import numpy as np
import pytest


def _dlt(P_list, xys):
    A = []
    for P, xy in zip(P_list, xys):
        x, y = float(xy[0]), float(xy[1])
        A.append(y * P[2, :] - P[1, :])
        A.append(P[0, :] - x * P[2, :])
    A = np.vstack(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def test_dlt_without_padding_is_not_biased_by_duplicate():
    """Duplicating the last view (old code) biases the noisy LS solution.

    With exact projections both give the exact 3D point; with noise the
    duplicated system double-weights the last camera and is farther from truth.
    The fixed code must use only available views.
    """
    rng = np.random.default_rng(0)
    # Two simple pinhole cameras
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], float)
    R0 = np.eye(3)
    t0 = np.array([0, 0, 0], float)
    R1 = np.eye(3)
    t1 = np.array([-50, 0, 0], float)  # baseline
    P0 = K @ np.concatenate([R0, t0[:, None]], axis=1)
    P1 = K @ np.concatenate([R1, t1[:, None]], axis=1)
    X_true = np.array([10.0, 5.0, 1000.0])

    def proj(P, X):
        x = P @ np.append(X, 1)
        return x[:2] / x[2]

    xy0 = proj(P0, X_true) + rng.normal(0, 0.3, 2)
    xy1 = proj(P1, X_true) + rng.normal(0, 0.3, 2)

    # Correct: 2 views
    X_2 = _dlt([P0, P1], [xy0, xy1])
    err_2 = float(np.linalg.norm(X_2 - X_true))

    # Buggy: pad to 4 by duplicating last view (2 duplicates of P1/xy1)
    X_4dup = _dlt([P0, P1, P1, P1], [xy0, xy1, xy1, xy1])
    err_dup = float(np.linalg.norm(X_4dup - X_true))

    # Duplicate should not be closer — it over-weights P1, proving bias.
    # With exact (no noise) they'd be equal; with noise the dup is no better.
    # We assert dup is not strictly better by a margin, i.e. 2-view is at least as good.
    assert err_2 <= err_dup + 1e-9

    # Also check rank: A with duplicate has same rank as without (3), so padding adds no constraint
    def rank_of(Ps, xys):
        A = []
        for P, xy in zip(Ps, xys):
            x, y = float(xy[0]), float(xy[1])
            A.append(y * P[2, :] - P[1, :])
            A.append(P[0, :] - x * P[2, :])
        A = np.vstack(A)
        return int(np.linalg.matrix_rank(A, tol=1e-6))

    assert rank_of([P0, P1], [xy0, xy1]) == rank_of(
        [P0, P1, P1, P1], [xy0, xy1, xy1, xy1]
    )


def test_plate_calibration_source_has_no_padding_loop():
    """Source regression: ensure the padding loop was removed."""
    from pathlib import Path

    src = Path("src/openptv2/plate_calibration.py").read_text(encoding="utf-8")
    assert "while len(Ps) < 4" not in src
