# ruff: noqa
"""Truth-free 2D streaks: per camera, link blobs frame to frame and describe every blob by its streak.

A streak is a chain of 1-1 links (Hungarian, gate in px). Where two particles merge, one chain ends next to the other
(or both end and a new one starts); where they separate, a chain starts next to another. Features per blob:
  qr    mass / 15th percentile of the streak's own mass within +-W frames (unit-mass reference)
  died  distance to the nearest other chain that ENDED within the L frames before this frame, near this chain
  born  distance to the nearest other chain that STARTED within the L frames after, near this chain
  nn    distance to the nearest other blob in the same frame
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

GATE = 2.0
R_TOPO = 3.0
L_TOPO = 6


def link_cam(B: dict, cam: int, f0: int, f1: int, gate: float = GATE) -> dict:
    """{(f, j): tracklet id}. ``B[(cam, f)]`` rows: [., x, y, n, nx, ny, sumg, .]"""
    tid, nxt, prev = {}, 0, None
    for f in range(f0, f1 + 1):
        rows = B[cam, f]
        m = {}
        if prev is not None and len(prev[0]) and len(rows):
            pr, pt = prev
            d = np.hypot(pr[:, None, 1] - rows[None, :, 1], pr[:, None, 2] - rows[None, :, 2])
            c = np.where(d <= gate, d, 1e6)
            ri, ci = linear_sum_assignment(c)
            m = {int(j): int(i) for i, j in zip(ri, ci) if c[i, j] < 1e5}
        cur = []
        for j in range(len(rows)):
            if j in m:
                cur.append(prev[1][m[j]])
            else:
                cur.append(nxt)
                nxt += 1
        for j, t in enumerate(cur):
            tid[f, j] = t
        prev = (rows, cur)
    return tid


def node_features(B: dict, cam: int, f0: int, f1: int, W: int = 30, gate: float = GATE) -> np.ndarray:
    """One row per blob: cam, f, j, n, sumg, qr, died, born, nn, tracklet length."""
    tid = link_cam(B, cam, f0, f1, gate)
    by = defaultdict(list)
    for (f, j), t in tid.items():
        by[t].append((f, j))
    pos = {}                                   # (t, f) -> (x, y, j)
    for t, nodes in by.items():
        nodes.sort()
        for f, j in nodes:
            pos[t, f] = (B[cam, f][j, 1], B[cam, f][j, 2], j)
    # death / birth events near other chains
    died_ev = defaultdict(list)                # t -> [(frame e, distance)]  another chain ended at e next to t
    born_ev = defaultdict(list)                # t -> [(frame s, distance)]  another chain started at s next to t
    trees = {}
    for f in range(f0, f1 + 1):
        r = B[cam, f]
        trees[f] = cKDTree(r[:, 1:3]) if len(r) else None
    inv = {f: np.array([tid[f, j] for j in range(len(B[cam, f]))]) for f in range(f0, f1 + 1)}
    for t, nodes in by.items():
        e = nodes[-1][0]                       # last frame of chain t
        if e < f1 and trees[e] is not None:
            x, y, _ = pos[t, e]
            for k in trees[e].query_ball_point([x, y], R_TOPO):
                o = inv[e][k]
                if o != t:
                    died_ev[o].append((e, float(np.hypot(B[cam, e][k, 1] - x, B[cam, e][k, 2] - y))))
        s = nodes[0][0]
        if s > f0 and trees[s] is not None:
            x, y, _ = pos[t, s]
            for k in trees[s].query_ball_point([x, y], R_TOPO):
                o = inv[s][k]
                if o != t:
                    born_ev[o].append((s, float(np.hypot(B[cam, s][k, 1] - x, B[cam, s][k, 2] - y))))
    nn = {}
    for f in range(f0, f1 + 1):
        r = B[cam, f]
        if len(r) > 1:
            d = trees[f].query(r[:, 1:3], k=2)[0][:, 1]
        else:
            d = np.full(len(r), 99.0)
        for j in range(len(r)):
            nn[f, j] = d[j]
    rows = []
    for t, nodes in by.items():
        fs = np.array([f for f, _ in nodes])
        ms = np.array([B[cam, f][j, 6] for f, j in nodes])
        for n, (f, j) in enumerate(nodes):
            w = np.abs(fs - f) <= W
            q = np.percentile(ms[w], 15) if w.sum() >= 3 else np.nan
            died = min([d for e, d in died_ev.get(t, []) if f - L_TOPO <= e < f], default=99.0)
            born = min([d for s, d in born_ev.get(t, []) if f < s <= f + L_TOPO], default=99.0)
            rows.append((cam, f, j, B[cam, f][j, 3], ms[n], ms[n] / q if np.isfinite(q) else np.nan, died, born, nn[f, j], len(nodes)))
    return np.array(rows)


def _fit_logit(X, y, l2=1.0, it=25):
    A = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(A.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-A @ w))
        g = A.T @ (p - y) + l2 * np.r_[0, w[1:]]
        H = (A * (p * (1 - p))[:, None]).T @ A + l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
        w -= np.linalg.solve(H + 1e-9 * np.eye(len(w)), g)
    return w


def cross_val_logit(X, y, groups, k=5):
    out = np.zeros(len(y))
    ug = np.unique(groups)
    for i in range(k):
        te = np.isin(groups, ug[i::k])
        w = _fit_logit(X[~te], y[~te].astype(float))
        out[te] = 1 / (1 + np.exp(-(np.column_stack([np.ones(te.sum()), X[te]]) @ w)))
    return out


def auc(s, y):
    r = np.argsort(np.argsort(s)) + 1
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


CUES = {
    "n": lambda D: np.log(D[:, 3]),
    "g": lambda D: np.log(D[:, 4]),
    "nn": lambda D: np.minimum(D[:, 8], 12),
    "qr": lambda D: np.log(np.nan_to_num(D[:, 5], nan=1.0)),
    "died": lambda D: np.exp(-np.minimum(D[:, 6], 8) ** 2 / 4),
    "born": lambda D: np.exp(-np.minimum(D[:, 7], 8) ** 2 / 4),
}


def ablate(D, y, groups, sets):
    """AUC and recall at 5 % false alarms for each cue set (5-fold by frame blocks)."""
    out = []
    for name, cols in sets:
        X = np.column_stack([CUES[c](D) for c in cols])
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        p = cross_val_logit(X, y, groups)
        thr = np.percentile(p[y == 0], 95)
        out.append((name, auc(p, y.astype(int)), (p[y == 1] > thr).mean()))
    return out


SETS = [
    ("blob alone (n, grey sum)", ["n", "g"]),
    ("+ nearest neighbour in this frame", ["n", "g", "nn"]),
    ("+ streak mass ratio", ["n", "g", "nn", "qr"]),
    ("+ streak topology (ended / started next to it)", ["n", "g", "nn", "died", "born"]),
    ("+ both", ["n", "g", "nn", "qr", "died", "born"]),
]
