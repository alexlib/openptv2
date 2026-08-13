"""JHTDB Lagrangian trajectory ingestion for the differentiable-PTV benchmark.

Queries the Johns Hopkins Turbulence Database (JHTDB) ``GetVelocity`` REST
service and integrates particle positions forward in time (RK4-free Euler
integration is sufficient at JHTDB's native ``dt``), giving ground-truth
Lagrangian trajectories from real DNS ($Re_\\lambda \\approx 433$ isotropic
turbulence).

The JHTDB service is occasionally unreachable (auth-token quota, outages, no
network egress in a sandbox). :func:`fetch_hit_trajectories` falls back to
:func:`synthetic_hit_trajectories` -- an Ornstein-Uhlenbeck velocity walk (the
same construction as :mod:`openptv2.benchmarking.scenario`'s
``flow_type="turbulent"``) -- so downstream pipeline stages always have a
ground-truth dataset to run against.
"""

from __future__ import annotations

import numpy as np
import requests

_JHTDB_VELOCITY_URL = "https://turbulence.pha.jhu.edu/service/turbulence.svc/GetVelocity"


def get_hit_velocity(
    token: str,
    points: np.ndarray,
    t: float = 0.0,
    dataset: str = "isotropic1024coarse",
    timeout: float = 15.0,
) -> np.ndarray:
    """Query one JHTDB velocity sample per point.

    Parameters
    ----------
    token : str
        JHTDB auth token (register at https://turbulence.pha.jhu.edu).
    points : ndarray (N, 3)
        Query points in the dataset's native domain (``[0, 2*pi)**3`` for
        ``isotropic1024coarse``).
    t : float
        Query time.
    dataset : str
        JHTDB dataset name.

    Returns
    -------
    ndarray (N, 3)
        Velocity at each point.

    Raises
    ------
    requests.RequestException
        On any network or HTTP error. Callers wanting an offline fallback
        should catch this (see :func:`fetch_hit_trajectories`).
    """
    out = np.zeros((len(points), 3))
    for i, (x, y, z) in enumerate(points):
        r = requests.get(
            _JHTDB_VELOCITY_URL,
            params=dict(
                authToken=token,
                dataset=dataset,
                t=t,
                spatialInterpolation="Lag6",
                spatialOperator="field",
                x=x,
                y=y,
                z=z,
            ),
            timeout=timeout,
        )
        r.raise_for_status()
        vals = [float(v) for v in r.text.strip().split(",")]
        out[i] = vals[:3]
    return out


def synthetic_hit_trajectories(
    n_particles: int,
    n_frames: int,
    dt: float = 0.002,
    domain: float = 80.0,
    seed: int = 42,
) -> np.ndarray:
    """Offline stand-in for JHTDB Lagrangian trajectories.

    Ornstein-Uhlenbeck velocity walk: smooth, chaotic, non-Gaussian-tailed
    enough to exercise the Stage-5 physics loss when JHTDB is unreachable.

    Returns
    -------
    ndarray (n_particles, n_frames, 3)
    """
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-domain / 2, domain / 2, size=(n_particles, 3))
    vel = np.zeros((n_particles, 3))
    traj = np.empty((n_particles, n_frames, 3))
    for f in range(n_frames):
        vel = 0.9 * vel + rng.normal(0.0, 1.0, size=(n_particles, 3))
        pos = pos + vel * dt
        traj[:, f, :] = pos
    return traj


def fetch_hit_trajectories(
    token: str,
    n_particles: int,
    n_frames: int,
    dt: float = 0.002,
    domain: float = 80.0,
    seed: int = 42,
    dataset: str = "isotropic1024coarse",
) -> tuple[np.ndarray, str]:
    """Fetch (or synthesize) ground-truth Lagrangian HIT trajectories.

    Tries JHTDB first (real DNS), falls back to
    :func:`synthetic_hit_trajectories` on any network/HTTP error.

    Returns
    -------
    trajectories : ndarray (n_particles, n_frames, 3)
    source : str
        ``"jhtdb"`` or ``"synthetic"``.
    """
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, 2 * np.pi, size=(n_particles, 3))
    traj = np.empty((n_particles, n_frames, 3))
    try:
        for f in range(n_frames):
            v = get_hit_velocity(token, pos, t=f * dt, dataset=dataset)
            pos = pos + v * dt
            traj[:, f, :] = pos
        return traj, "jhtdb"
    except requests.RequestException:
        return synthetic_hit_trajectories(n_particles, n_frames, dt, domain, seed), "synthetic"


__all__ = [
    "get_hit_velocity",
    "synthetic_hit_trajectories",
    "fetch_hit_trajectories",
]
