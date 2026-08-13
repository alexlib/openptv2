"""Sobol global sensitivity analysis (Phase 4).

Causal attribution between Stage-1..4 micro-parameters and the Stage-5
Lagrangian physics loss: which upstream parameters actually move the
downstream physics, via Saltelli's (2010) improved Sobol estimator.

Uses ``scipy.stats.qmc.Sobol`` for the quasi-random base sequence -- no new
dependency (SALib) needed. Validated against the analytic Sobol indices of a
purely additive linear test function before landing (S1 == ST == exact
values for that case).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc


def saltelli_sample(
    bounds: list[tuple[float, float]], n_base: int, seed: int | None = 0
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Generate Saltelli-scheme samples for Sobol sensitivity analysis.

    Parameters
    ----------
    bounds : list of (low, high)
        One range per parameter.
    n_base : int
        Base sample size N.

    Returns
    -------
    A, B : ndarray (N, k)
    AB : list[ndarray] of length k
        Each entry is ``A`` with column ``i`` replaced by ``B``'s column ``i``.
    """
    k = len(bounds)
    sampler = qmc.Sobol(d=2 * k, scramble=True, seed=seed)
    unit = sampler.random(n_base)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    A = lo + unit[:, :k] * (hi - lo)
    B = lo + unit[:, k:] * (hi - lo)
    AB = []
    for i in range(k):
        M = A.copy()
        M[:, i] = B[:, i]
        AB.append(M)
    return A, B, AB


def sobol_indices(
    objective, bounds: list[tuple[float, float]], n_base: int = 256, seed: int | None = 0
) -> dict[str, np.ndarray]:
    """First-order and total-order Sobol sensitivity indices (Saltelli 2010).

    Parameters
    ----------
    objective : callable
        ``(N, k) ndarray -> (N,) ndarray`` of scalar outputs (e.g. the total
        physics loss evaluated at each parameter sample).
    bounds : list of (low, high)
        One range per parameter.
    n_base : int
        Base sample size; total evaluations = ``n_base * (k + 2)``.

    Returns
    -------
    dict
        ``"S1"`` (k,) first-order indices, ``"ST"`` (k,) total-order indices.
    """
    A, B, AB = saltelli_sample(bounds, n_base, seed=seed)
    f_A = objective(A)
    f_B = objective(B)
    f_AB = [objective(M) for M in AB]
    var_y = np.concatenate([f_A, f_B]).var()

    k = len(bounds)
    s1 = np.empty(k)
    st = np.empty(k)
    for i in range(k):
        s1[i] = np.mean(f_B * (f_AB[i] - f_A)) / var_y
        st[i] = 0.5 * np.mean((f_A - f_AB[i]) ** 2) / var_y
    return {"S1": s1, "ST": st}


__all__ = ["saltelli_sample", "sobol_indices"]
