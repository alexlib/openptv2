"""
Gaussian Mixture Model probabilistic track approximation (vendored from proPTV).

Original source: https://github.com/RobinBarta/proPTV

MIT License
Copyright (c) 2023 DLR (Project owner: Robin Barta)

This module vendors/adapts parts of the proPTV framework. The above
copyright notice and this permission notice are included in all copies or
substantial portions of this Software, and the underlying publication must
be cited:

  Barta, Robin, et al. "proPTV - A probabilistic particle tracking
  velocimetry framework." Journal of Computational Physics (2024).
  https://doi.org/10.1016/j.jcp.2024.113212
"""

from __future__ import annotations

import numpy as np
import numpy.matlib as ml


def GMM(t, X):
    """Fit Gaussian Mixture Model to track time-position history.

    Parameters
    ----------
    t : ndarray (N,)
        Time steps.
    X : ndarray (N, 3)
        Particle positions.

    Returns
    -------
    w : ndarray (N, 3)
        Weights of Gaussian basis functions.
    psi_X : ndarray (N, N)
        Position basis functions.
    psi_V : ndarray (N, N)
        Velocity basis functions.
    psi_A : ndarray (N, N)
        Acceleration basis functions.
    """
    N = len(t)
    centers = np.array([np.linspace(0 - 3, 1 + 3, N)])
    kernel_size = (centers[0, 1] - centers[0, 0]) ** 2
    z = np.linspace(0, 1, len(t))

    x = ml.repmat(z, N, 1) - ml.repmat(centers.T, 1, len(t))
    b = np.exp(-(x**2) / (2 * kernel_size))
    b_dt = b * (-x / kernel_size)
    b_dt_dt = (b * (-x / kernel_size) ** 2) - (b / kernel_size)

    sum_b = ml.repmat(np.sum(b, axis=0), N, 1)
    sum_b_dt = ml.repmat(np.sum(b_dt, axis=0), N, 1)
    sum_b_dt_dt = ml.repmat(np.sum(b_dt_dt, axis=0), N, 1)

    psi_X = (b / sum_b).T
    psi_V = (
        (b_dt * sum_b - b * sum_b_dt) / (sum_b**2) * (z[1] - z[0])
    ).T
    psi_A = (
        (
            ((b_dt_dt * sum_b - b * sum_b_dt_dt) * sum_b**2)
            - ((b_dt * sum_b - b * sum_b_dt) * (2 * sum_b * sum_b_dt))
        )
        / (sum_b**4)
        * (z[1] - z[0]) ** 2
    ).T

    w = np.linalg.solve(
        psi_X.T @ psi_X + np.eye(psi_X.shape[1]) * 1e-10,
        psi_X.T @ X,
    )
    return w, psi_X, psi_V, psi_A


def Approximate(t, w, psi_X, psi_V, psi_A):
    """Evaluate the GMM approximation at time steps t."""
    X = psi_X @ w
    V = psi_V @ w
    A = psi_A @ w
    return X, V, A


def Predict(t, X, V, A):
    """Predict next state from GMM approximation."""
    dt_sign = np.sign(np.diff(t)[0])
    return (
        X[-1] + V[-1] * dt_sign,
        V[-1] + A[-1] * dt_sign,
        A[-1] + (A[-1] - A[-2]),
    )
