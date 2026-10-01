"""Reconnect broken trajectory pieces with smooth-curve guesses (plan A5).

Arrays follow the pred.npz layout: ``trajid``, ``frame``, ``pos`` per point, any order.
A piece END is joined to a later piece START when a low-order polynomial (a straight line by default: measured better than a quadratic, which amplifies the noise when extrapolated) fitted to the
last points of the first piece, extrapolated forward, and one fitted to the first points
of the second piece, extrapolated backward, both land within a tolerance that follows the
position noise and grows with the gap. Joins are made best first, each end and each start
used once. Nothing is invented: points are only relabelled.
"""

from __future__ import annotations

import numpy as np


def _fit(t, x, deg):
    c = np.polynomial.polynomial.polyfit(t, x, deg)
    return c


def _eval(c, t):
    return np.polynomial.polynomial.polyval(t, c).T if c.ndim > 1 else None


def reconnect_pieces(
    trajid: np.ndarray,
    frame: np.ndarray,
    pos: np.ndarray,
    max_gap: int = 4,
    tol_sigmas: float = 4.0,
    n_fit: int = 10,
    deg: int = 1,
    min_len: int = 4,
) -> np.ndarray:
    """New trajectory labels (same order as the input) after reconnecting.

    ``max_gap`` is the largest frame step between the end of one piece and the start of
    the next (1 = adjacent frames). The tolerance for a step ``g`` is
    ``tol_sigmas * sigma * (1 + 0.5 * (g - 1))``, ``sigma`` being the median residual
    scatter (mm) of the polynomial fits over all pieces."""
    order = np.lexsort((frame, trajid))
    tid, fr, ps = trajid[order], frame[order], pos[order].astype(np.float64)
    cuts = np.flatnonzero(np.diff(tid)) + 1
    groups = np.split(np.arange(len(order)), cuts)
    pieces = [g for g in groups if len(g) >= min_len]
    if not pieces:
        return trajid.copy()

    def fit_end(g, tail):
        m = min(n_fit, len(g))
        idx = g[-m:] if tail else g[:m]
        t = fr[idx].astype(float)
        t0 = t[-1] if tail else t[0]
        d = min(deg, m - 2)
        c = np.stack(
            [np.polynomial.polynomial.polyfit(t - t0, ps[idx, k], d) for k in range(3)]
        )
        res = ps[idx] - np.stack(
            [np.polynomial.polynomial.polyval(t - t0, c[k]) for k in range(3)], axis=1
        )
        return c, t0, float(np.sqrt(np.sum(res**2) / max(m - d - 1, 1) / 3.0))

    ends = [fit_end(g, True) for g in pieces]
    starts = [fit_end(g, False) for g in pieces]
    sigma = max(float(np.median([e[2] for e in ends] + [s[2] for s in starts])), 1e-6)

    last_f = np.array([fr[g[-1]] for g in pieces])
    first_f = np.array([fr[g[0]] for g in pieces])
    last_p = np.array([ps[g[-1]] for g in pieces])
    first_p = np.array([ps[g[0]] for g in pieces])

    cand = []
    order_start = np.argsort(first_f)
    sf = first_f[order_start]
    for i in range(len(pieces)):
        lo = np.searchsorted(sf, last_f[i] + 1, "left")
        hi = np.searchsorted(sf, last_f[i] + max_gap, "right")
        for j in order_start[lo:hi]:
            if j == i:
                continue
            g = int(first_f[j] - last_f[i])
            ce, te, _ = ends[i]
            cs, ts, _ = starts[j]
            fwd = np.array(
                [
                    np.polynomial.polynomial.polyval(first_f[j] - te, ce[k])
                    for k in range(3)
                ]
            )
            bwd = np.array(
                [
                    np.polynomial.polynomial.polyval(last_f[i] - ts, cs[k])
                    for k in range(3)
                ]
            )
            e1 = np.linalg.norm(fwd - first_p[j])
            e2 = np.linalg.norm(bwd - last_p[i])
            tol = tol_sigmas * sigma * (1 + 0.5 * (g - 1))
            if e1 < tol and e2 < tol:
                cand.append(((e1 + e2) / tol, i, int(j)))
    cand.sort()
    used_end, used_start = set(), set()
    parent = list(range(len(pieces)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for _, i, j in cand:
        if i in used_end or j in used_start or find(i) == find(j):
            continue
        used_end.add(i)
        used_start.add(j)
        parent[find(j)] = find(i)

    new_tid = tid.copy()
    for k, g in enumerate(pieces):
        new_tid[g] = tid[pieces[find(k)][0]]
    out = np.empty_like(trajid)
    out[order] = new_tid
    return out
