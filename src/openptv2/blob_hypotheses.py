"""Per-blob alternatives for the detector: is this blob one tracer or two overlapping ones?

Works on the high-passed image and the targets the pipeline already found. It never changes the targets; it
returns, for every blob, shape measures and a best two-tracer explanation that later steps may use at conflict
sites (docs/plans/2026-11-21-detection-hypotheses-and-events.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURES = ("n", "mean_int", "sxx", "sxy", "syy", "lam1", "lam2", "ecc", "n_max", "m3")


def blob_features(px: np.ndarray, thres: float):
    """Shape and brightness measures of one blob from its own pixels ``(x, y, grey)`` (pixel centres at integer
    coordinates; the targets add +0.5). Weights are ``grey - thres``, as in the target centre."""
    x, y, g = px[:, 0], px[:, 1], px[:, 2]
    w = np.clip(g - thres, 0, None)
    tot = w.sum()
    if tot <= 0 or len(px) < 2:
        return None
    cx, cy = (w * x).sum() / tot, (w * y).sum() / tot
    dx, dy = x - cx, y - cy
    sxx, syy, sxy = (w * dx * dx).sum() / tot, (w * dy * dy).sum() / tot, (w * dx * dy).sum() / tot
    lam, vec = np.linalg.eigh(np.array([[sxx, sxy], [sxy, syy]]))     # ascending
    proj = dx * vec[0, 1] + dy * vec[1, 1]
    m3 = (w * proj**3).sum() / tot
    # internal local maxima: pixels not lower than any of their 8 neighbours inside the blob
    grid = {(int(a), int(b)): v for a, b, v in zip(x, y, g)}
    n_max = 0
    for (a, b), v in grid.items():
        if all(grid.get((a + i, b + j), -1.0) <= v for i in (-1, 0, 1) for j in (-1, 0, 1) if i or j):
            n_max += 1
    ecc = float(np.sqrt(max(1.0 - lam[0] / lam[1], 0.0))) if lam[1] > 0 else 0.0
    n = float(len(px))
    return (n, float(g.sum()) / n, sxx, sxy, syy, lam[1], lam[0], ecc, float(n_max), m3)


def _gauss(xx, yy, x0, y0, amp, s2):
    return amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * s2))


def fit_one(img: np.ndarray, px: np.ndarray, s2: float, margin: int = 1):
    """One Gaussian of fixed variance ``s2`` fitted to the blob's bounding window. Returns ``(x, y, amp, cost)``
    in integer-pixel-centre coordinates."""
    from scipy.optimize import least_squares

    xx, yy, g = _fit_window(img, px, margin)
    c0 = ((px[:, 0] * px[:, 2]).sum() / px[:, 2].sum(), (px[:, 1] * px[:, 2]).sum() / px[:, 2].sum())

    def res(p):
        return _gauss(xx, yy, p[0], p[1], p[2], s2) - g

    r = least_squares(res, [c0[0], c0[1], float(px[:, 2].max())], method="lm")
    return r.x[0], r.x[1], r.x[2], float((r.fun ** 2).sum())


def fit_two(img: np.ndarray, px: np.ndarray, s2: float, margin: int = 1):
    """Two Gaussians of fixed variance ``s2`` (separate amplitudes) fitted to the blob's window, started from the
    blob's principal axis. Returns ``(x1, y1, a1, x2, y2, a2, cost)``."""
    from scipy.optimize import least_squares

    xx, yy, g = _fit_window(img, px, margin)
    w = np.clip(px[:, 2], 1e-9, None)
    cx, cy = (px[:, 0] * w).sum() / w.sum(), (px[:, 1] * w).sum() / w.sum()
    cov = np.cov(np.vstack([px[:, 0] - cx, px[:, 1] - cy]), aweights=w)
    lam, vec = np.linalg.eigh(cov)
    d = max(np.sqrt(max(lam[1] - s2, 0.0)), 0.4)
    ax = vec[:, 1]
    amp = float(px[:, 2].max())

    def res(p):
        return (_gauss(xx, yy, p[0], p[1], p[2], s2) + _gauss(xx, yy, p[3], p[4], p[5], s2)) - g

    best = None
    for sgn in (1.0,):
        p0 = [cx - sgn * d * ax[0], cy - sgn * d * ax[1], amp * 0.6, cx + sgn * d * ax[0], cy + sgn * d * ax[1], amp * 0.6]
        r = least_squares(res, p0, method="lm")
        c = float((r.fun ** 2).sum())
        if best is None or c < best[-1]:
            best = (*r.x, c)
    return best


def _fit_window(img: np.ndarray, px: np.ndarray, margin: int):
    h, w = img.shape
    x0, x1 = max(int(px[:, 0].min()) - margin, 0), min(int(px[:, 0].max()) + margin + 1, w)
    y0, y1 = max(int(px[:, 1].min()) - margin, 0), min(int(px[:, 1].max()) + margin + 1, h)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return xx.astype(float).ravel(), yy.astype(float).ravel(), img[y0:y1, x0:x1].astype(float).ravel()


@dataclass
class Logistic:
    """Small L2-regularised logistic regression (Newton steps); features are standardised inside."""
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    bias: float

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> "Logistic":
        mean, scale = X.mean(0), X.std(0) + 1e-9
        Z = np.hstack([(X - mean) / scale, np.ones((len(X), 1))])
        w = np.zeros(Z.shape[1])
        reg = l2 * np.eye(Z.shape[1])
        reg[-1, -1] = 0.0
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-Z @ w))
            g = Z.T @ (p - y) + reg @ w
            H = (Z * (p * (1 - p))[:, None]).T @ Z + reg + 1e-9 * np.eye(Z.shape[1])
            step = np.linalg.solve(H, g)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        return cls(mean, scale, w[:-1], float(w[-1]))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(((X - self.mean) / self.scale) @ self.coef + self.bias)))
