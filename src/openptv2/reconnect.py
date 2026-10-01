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


def _end_fits(frame, pos, groups, n_fit, deg, tail):
    """Batched polynomial fits of the last (tail) or first points of every piece.

    Returns coefficients (P, 3, deg+1) in powers of (frame - reference frame), the
    reference frame (P,) and the residual scatter (P,) in mm. Pieces are grouped by the
    number of points used so each group is one batched least-squares solve."""
    n_p = len(groups)
    coef = np.zeros((n_p, 3, deg + 1))
    ref = np.zeros(n_p)
    sig = np.zeros(n_p)
    m_of = np.array([min(n_fit, len(g)) for g in groups])
    for m in np.unique(m_of):
        sel = np.flatnonzero(m_of == m)
        d = int(min(deg, m - 2))
        idx = np.stack([groups[k][-m:] if tail else groups[k][:m] for k in sel])
        t = frame[idx].astype(float)
        t0 = t[:, -1] if tail else t[:, 0]
        dt = t - t0[:, None]
        A = dt[:, :, None] ** np.arange(d + 1)[None, None, :]  # (S, m, d+1)
        y = pos[idx]  # (S, m, 3)
        AtA = A.transpose(0, 2, 1) @ A
        c = np.linalg.solve(AtA + 1e-12 * np.eye(d + 1), A.transpose(0, 2, 1) @ y)
        res = y - A @ c
        coef[sel, :, : d + 1] = c.transpose(0, 2, 1)
        ref[sel] = t0
        sig[sel] = np.sqrt((res**2).sum(axis=(1, 2)) / max(m - d - 1, 1) / 3.0)
    return coef, ref, sig


def _poly(coef, dt):
    """coef (K, 3, deg+1), dt (K,) -> (K, 3) values."""
    powers = dt[:, None] ** np.arange(coef.shape[2])[None, :]
    return np.einsum("kcp,kp->kc", coef, powers)


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
    pieces = [g for g in np.split(np.arange(len(order)), cuts) if len(g) >= min_len]
    if not pieces:
        return trajid.copy()

    ce, te, se = _end_fits(fr, ps, pieces, n_fit, deg, tail=True)
    cs, ts, ss = _end_fits(fr, ps, pieces, n_fit, deg, tail=False)
    sigma = max(float(np.median(np.concatenate([se, ss]))), 1e-6)

    first_i = np.array([g[0] for g in pieces])
    last_i = np.array([g[-1] for g in pieces])
    first_f, last_f = fr[first_i], fr[last_i]
    first_p, last_p = ps[first_i], ps[last_i]

    # candidate pairs (end of i, start of j) with 1 <= start - end <= max_gap
    by_start = np.argsort(first_f, kind="stable")
    sf = first_f[by_start]
    lo = np.searchsorted(sf, last_f + 1, "left")
    hi = np.searchsorted(sf, last_f + max_gap, "right")
    counts = hi - lo
    ii = np.repeat(np.arange(len(pieces)), counts)
    offs = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    jj = by_start[np.repeat(lo, counts) + offs]
    keep = jj != ii
    ii, jj = ii[keep], jj[keep]

    gap = first_f[jj] - last_f[ii]
    fwd = _poly(ce[ii], first_f[jj] - te[ii])
    bwd = _poly(cs[jj], last_f[ii] - ts[jj])
    e1 = np.linalg.norm(fwd - first_p[jj], axis=1)
    e2 = np.linalg.norm(bwd - last_p[ii], axis=1)
    tol = tol_sigmas * sigma * (1 + 0.5 * (gap - 1))
    ok = (e1 < tol) & (e2 < tol)
    ii, jj = ii[ok], jj[ok]
    cost = ((e1 + e2) / tol)[ok]
    # best first; ties by the order the pairs were generated (as a stable sort does)
    o = np.argsort(cost, kind="stable")

    used_end = np.zeros(len(pieces), bool)
    used_start = np.zeros(len(pieces), bool)
    parent = np.arange(len(pieces))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in zip(ii[o].tolist(), jj[o].tolist()):
        if used_end[i] or used_start[j] or find(i) == find(j):
            continue
        used_end[i] = used_start[j] = True
        parent[find(j)] = find(i)

    root = np.array([find(k) for k in range(len(pieces))])
    new_tid = tid.copy()
    for k, g in enumerate(pieces):
        new_tid[g] = tid[pieces[root[k]][0]]
    out = np.empty_like(trajid)
    out[order] = new_tid
    return out
