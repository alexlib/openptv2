"""Pixel-level de-merging: measure the particles of a shared blob from the image pixels.

A blob shared by ``m`` tracked particles is fitted with ``m`` Gaussian spots (free amplitudes, one common width, a local
plane as background), each spot seeded at the track's predicted pixel position and held there by a Gaussian prior of the
prediction's own error. The result replaces the centre-split observation of :func:`openptv2.blob_model.split_blobs_effective`
in :func:`openptv2.merge_refine.refine_merged`. Other blobs near the window are fitted along as spots with a loose prior,
so their light is not given to the members.

Validated on a rendered case with truth (plan 2026-11-21-tracker-setup.md, section 18): merged members, median error
0.15 px and 2 to 3 % above 1 px, against 0.35 px and 9 % for the centre split.

The pixel noise is not assumed: it is read from the residual of a first fit with a loose prior (photon noise and model
misfit included), and the prior is weighted against it. The reported variance comes from the same fit.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable

import numpy as np
from scipy.optimize import least_squares

SIGMA0 = 0.75            # starting width of a spot (px); the fit frees it within SIGMA_RANGE
SIGMA_RANGE = (0.5, 1.2)
MIN_WINDOW, MAX_WINDOW = 6, 13
OTHER_PRIOR_PX = 1.0     # a neighbouring blob's centre may itself be shifted by a merge
MAX_MOVE_PX = 3.0        # a member that moves further than this from its seed is not trusted


def _spots(xx: np.ndarray, yy: np.ndarray, p: np.ndarray, sigma: float) -> np.ndarray:
    out = np.zeros(xx.shape)
    for j in range(len(p) // 3):
        x, y, a = p[3 * j: 3 * j + 3]
        out += a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma ** 2))
    return out


def fit_window(
    img: np.ndarray,
    seeds: np.ndarray,
    prior_px: np.ndarray,
    centre: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Fit the spots at ``seeds`` (K, 2; pixel-centre convention of ``targ_rec``: pixel (i, j) is centred at i + .5)
    in the window of half-size ``radius`` around ``centre``. ``prior_px`` (K,) is the prior sigma of each spot's position.

    Returns ``(positions (K, 2), variances (K,) in px^2 per axis, residual rms)`` or ``None`` when the window leaves the
    image or the fit fails.
    """
    h, w = img.shape
    x0, y0 = int(np.floor(centre[0])) - radius, int(np.floor(centre[1])) - radius
    side = 2 * radius + 1
    if x0 < 0 or y0 < 0 or x0 + side > w or y0 + side > h:
        return None
    win = img[y0:y0 + side, x0:x0 + side].astype(float)
    yy, xx = np.mgrid[y0:y0 + side, x0:x0 + side] + 0.5
    k = len(seeds)
    bg = float(np.median(win))
    p0 = np.empty(3 * k)
    for j, (sx, sy) in enumerate(seeds):
        ix = int(np.clip(np.floor(sx - xx[0, 0] + 0.5), 0, side - 1))
        iy = int(np.clip(np.floor(sy - yy[0, 0] + 0.5), 0, side - 1))
        p0[3 * j: 3 * j + 3] = (sx, sy, max(float(win[iy, ix]) - bg, 5.0))
    cx, cy = xx.mean(), yy.mean()
    start = np.concatenate([p0, [SIGMA0, bg, 0.0, 0.0]])
    lo = np.concatenate([np.tile([-np.inf, -np.inf, 0.0], k), [SIGMA_RANGE[0], -np.inf, -np.inf, -np.inf]])
    hi = np.concatenate([np.tile([np.inf, np.inf, np.inf], k), [SIGMA_RANGE[1], np.inf, np.inf, np.inf]])
    npix = win.size

    def model(q):
        return (_spots(xx, yy, q[:3 * k], q[3 * k]) + q[3 * k + 1] + q[3 * k + 2] * (xx - cx)
                + q[3 * k + 3] * (yy - cy))

    def resid(q, noise):
        r = (model(q) - win).ravel()
        pr = np.empty(2 * k)
        pr[0::2] = (q[0:3 * k:3] - p0[0::3]) * noise / prior_px
        pr[1::2] = (q[1:3 * k:3] - p0[1::3]) * noise / prior_px
        return np.concatenate([r, pr])

    try:
        # 1) loose prior: only to read the noise level of this window
        first = least_squares(resid, start, bounds=(lo, hi), args=(1.0,), method="trf", max_nfev=40)
        noise = float(np.sqrt(np.sum(first.fun[:npix] ** 2) / max(npix - (3 * k + 4), 1)))
        noise = max(noise, 1.0)
        # 2) prior weighted against that noise
        sol = least_squares(resid, start, bounds=(lo, hi), args=(noise,), method="trf", max_nfev=40)
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not sol.success and sol.status != 0:
        return None
    jac = sol.jac
    try:
        cov = noise ** 2 * np.linalg.inv(jac.T @ jac + 1e-12 * np.eye(jac.shape[1]))
    except np.linalg.LinAlgError:
        return None
    pos = np.stack([sol.x[0:3 * k:3], sol.x[1:3 * k:3]], axis=1)
    var = np.array([0.5 * (cov[3 * j, 3 * j] + cov[3 * j + 1, 3 * j + 1]) for j in range(k)])
    amp = sol.x[2:3 * k:3]
    var = np.where(amp > 3.0 * noise, var, np.nan)          # a spot not above the noise carries no measurement
    return pos, var, noise


class PixelFitter:
    """Callable used by ``refine_merged(pixel_fit=...)``.

    ``read_images(frame)`` returns the per-camera images of a frame (``read_frame_images`` of the run); the last few
    frames stay in memory, since the groups of one camera come in frame order.
    """

    def __init__(self, read_images: Callable[[int], list], cache_frames: int = 3, min_var: float = 0.01,
                 var_scale: float = 2.5, min_prior_px: float = 0.15):
        self.read_images = read_images
        self.cache_frames = cache_frames
        self.min_var = float(min_var)
        self.min_prior_px = float(min_prior_px)  # floor of the prior sigma (the track prediction's error in the image)
        self.var_scale = float(var_scale)      # measured on the rendered case: the covariance is 2.6x too small (model misfit)
        self._cache: OrderedDict[int, list] = OrderedDict()
        self.stats = {"groups": 0, "fitted": 0, "failed": 0, "noise": []}

    def _image(self, frame: int, cam: int) -> np.ndarray:
        if frame in self._cache:
            self._cache.move_to_end(frame)
        else:
            self._cache[frame] = self.read_images(int(frame))
            while len(self._cache) > self.cache_frames:
                self._cache.popitem(last=False)
        return self._cache[frame][cam]

    def __call__(self, frame: int, cam: int, centre: np.ndarray, seeds: np.ndarray, prior_px: float,
                 others: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray] | None:
        """One shared blob: ``centre`` (2,), member predictions ``seeds`` (m, 2), the prediction error ``prior_px``,
        ``others`` (K, 2) centres of the other blobs of this frame and camera. Returns ``(positions (m, 2),
        variances (m,))`` (variance NaN where a member could not be measured) or ``None``."""
        self.stats["groups"] += 1
        m = len(seeds)
        spread = float(np.max(np.hypot(*(seeds - centre).T))) if m else 0.0
        radius = int(np.clip(np.ceil(spread) + 4, MIN_WINDOW, MAX_WINDOW))
        spots, priors = [seeds], [np.full(m, prior_px)]
        if others is not None and len(others):
            d = np.hypot(others[:, 0] - centre[0], others[:, 1] - centre[1])
            near = (d > 0.3) & (d < radius + 1.5)
            if near.any():
                spots.append(others[near])
                priors.append(np.full(int(near.sum()), OTHER_PRIOR_PX))
        out = fit_window(self._image(frame, cam), np.vstack(spots), np.concatenate(priors), centre, radius)
        if out is None:
            self.stats["failed"] += 1
            return None
        pos, var, noise = out
        pos, var = pos[:m], var[:m]
        moved = np.hypot(*(pos - seeds).T) > MAX_MOVE_PX
        var = np.where(moved, np.nan, np.maximum(self.var_scale * var, self.min_var))
        self.stats["fitted"] += 1
        self.stats["noise"].append(noise)
        return pos, var
