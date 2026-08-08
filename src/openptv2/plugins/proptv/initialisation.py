"""
Track initialisation routines (vendored from proPTV).

Original source: https://github.com/RobinBarta/proPTV
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def init_position_3d(track):
    """Savitzky-Golay smoothed positions."""
    pos = np.zeros_like(track)
    w = min(len(track), 5)
    p = min(len(track) - 1, 3)
    for d in range(3):
        pos[:, d] = signal.savgol_filter(track[:, d], w, p, deriv=0, mode="interp")
    return pos


def init_velocity_3d(track):
    """Savitzky-Golay first-derivative velocity estimates."""
    vel = np.zeros_like(track)
    w = min(len(track), 5)
    p = min(len(track) - 1, 3)
    for d in range(3):
        vel[:, d] = signal.savgol_filter(track[:, d], w, p, deriv=1, mode="interp")
    return vel


def init_acceleration_3d(track):
    """Savitzky-Golay second-derivative acceleration estimates."""
    acc = np.zeros_like(track)
    w = min(len(track), 5)
    p = min(len(track) - 1, 3)
    for d in range(3):
        acc[:, d] = signal.savgol_filter(track[:, d], w, p, deriv=2, mode="interp")
    return acc


def find_nn_points(p, P, N):
    """Find N nearest neighbour points in P around p."""
    return P[np.argpartition(np.linalg.norm(p - P, axis=1), N)[:N]]
