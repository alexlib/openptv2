"""Light check: does the raw image show a particle where a 3-camera point says it must be?

A point matched in all cameras but one is the weakest kind the stereo matcher makes: two or three rays that meet by
chance (a ghost) look like a particle seen in one camera less. A real particle, however, leaves light in the camera
that did not see it, unless it is very faint or the blob there was used by a neighbour. So the point is projected into
the missing camera and the raw image is read at that place: the peak within ``R_IN`` px over the local background,
in units of the local noise (``z``). A ghost shows noise there (median z about 4), a real particle a clear peak
(median z about 90 on the rendered benchmark; about half that on the real recording wp2).

Only points with exactly one camera missing are judged. Points whose projection falls too close to the image edge
for the window are kept: nothing is known about them. Validation: docs/plans/2026-11-21-tracker-setup.md, section 23.

``ptv.light_check`` in the parameter file switches the step on (``true`` = ``DEFAULT_Z_MIN``, or a number).
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np

#: a 3-camera point is kept when the light in its missing view reaches this many noise units
DEFAULT_Z_MIN = 6.0
#: radius (px) within which the peak is taken, window half-size, and the radius beyond which pixels are background
R_IN = 1.5
R_WIN = 7
R_RING = 4.0


def light_check_from_params(ptv: dict | None) -> float | None:
    """The ``light_check`` threshold of the ``ptv`` section: ``None`` (off) for missing, false or 0, the default for
    true, else the number given."""
    value = (ptv or {}).get("light_check", None)
    if value is None or value is False or value == 0:
        return None
    return DEFAULT_Z_MIN if value is True else float(value)


def light_z(img: np.ndarray, xy: np.ndarray, r_in: float = R_IN, r_win: int = R_WIN, r_ring: float = R_RING) -> np.ndarray:
    """Peak of the image within ``r_in`` px of each point ``xy`` (n, 2) (pixel-centre convention: pixel (i, j) is
    centred at i + 0.5, as in ``targ_rec``), over the local background, in units of the local noise.

    Background = median and noise = 1.4826 x MAD of the pixels farther than ``r_ring`` from the point inside the
    square window of half-size ``r_win``. NaN where the window leaves the image or the point is not finite.
    """
    xy = np.asarray(xy, float).reshape(-1, 2)
    out = np.full(len(xy), np.nan)
    h, w = img.shape
    fin = np.isfinite(xy).all(axis=1)
    x0 = np.floor(np.where(fin, xy[:, 0], 0.0)).astype(int)
    y0 = np.floor(np.where(fin, xy[:, 1], 0.0)).astype(int)
    ok = fin & (x0 - r_win >= 0) & (y0 - r_win >= 0) & (x0 + r_win < w) & (y0 + r_win < h)
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return out
    off = np.arange(-r_win, r_win + 1)
    ox, oy = (a.ravel() for a in np.meshgrid(off, off))
    xs = x0[idx, None] + ox[None, :]
    ys = y0[idx, None] + oy[None, :]
    win = img[ys, xs].astype(float)
    d = np.hypot(xs + 0.5 - xy[idx, 0, None], ys + 0.5 - xy[idx, 1, None])
    ring = np.where(d > r_ring, win, np.nan)
    bg = np.nanmedian(ring, axis=1)
    sd = 1.4826 * np.nanmedian(np.abs(ring - bg[:, None]), axis=1) + 1.0
    peak = np.where(d <= r_in, win, -np.inf).max(axis=1)
    out[idx] = (peak - bg) / sd
    return out


def _project(pos: np.ndarray, cal: Any, cpar: Any) -> np.ndarray:
    from openptv2.loco import project_pixels

    return project_pixels(pos, cal, cpar)


def missing_view_light(
    pos: np.ndarray,
    ids: np.ndarray,
    images: list[np.ndarray],
    cals: list,
    cpar: Any,
    project: Callable[[np.ndarray, Any, Any], np.ndarray] | None = None,
) -> np.ndarray:
    """``z`` of the missing view for the points with exactly one camera missing (NaN for all other points and for
    projections too near the image edge). ``pos`` (n, 3) mm, ``ids`` (n, >= n_cams) blob ids with -1 = not seen,
    ``images`` one image per camera. ``project(points, cal, cpar)`` returns pixels (default: the camera model)."""
    project = project or _project
    n_cams = len(cals)
    ids = np.asarray(ids)[:, :n_cams]
    pos = np.asarray(pos, float).reshape(-1, 3)
    z = np.full(len(pos), np.nan)
    if n_cams < 4 or len(pos) == 0:
        return z
    missing = ids < 0
    one = missing.sum(axis=1) == 1
    for c in range(n_cams):
        rows = np.flatnonzero(one & missing[:, c])
        if len(rows):
            z[rows] = light_z(images[c], project(pos[rows], cals[c], cpar))
    return z


def light_keep(
    pos: np.ndarray,
    ids: np.ndarray,
    images: list[np.ndarray],
    cals: list,
    cpar: Any,
    z_min: float = DEFAULT_Z_MIN,
    project: Callable[[np.ndarray, Any, Any], np.ndarray] | None = None,
) -> np.ndarray:
    """Boolean mask of the points to keep: every point except the one-camera-missing points whose missing view
    shows less than ``z_min`` noise units of light."""
    z = missing_view_light(pos, ids, images, cals, cpar, project)
    return ~(np.isfinite(z) & (z < z_min))


def apply_light_check(
    store: Any,
    frames: list[int],
    read_images: Callable[[int], list[np.ndarray]],
    cals: list,
    cpar: Any,
    z_min: float = DEFAULT_Z_MIN,
    project: Callable[[np.ndarray, Any, Any], np.ndarray] | None = None,
) -> dict[int, tuple[int, int]]:
    """Filter the stored correspondences of ``frames`` with :func:`light_keep` and write them back (the per-point
    ``rcm`` array, when present, is filtered with them to stay row-aligned). ``read_images(frame)`` returns the
    per-camera images. A frame whose images cannot be read is left as it is (all points kept) and reported in one
    warning at the end: the check never aborts a run. Returns ``{frame: (points kept, points before)}``."""
    stats: dict[int, tuple[int, int]] = {}
    skipped: list[tuple[int, str]] = []
    for f in frames:
        if not store.has_correspondences(f):
            continue
        pos, ids = store.read_correspondences(f)
        pos, ids = np.asarray(pos), np.asarray(ids)
        if len(pos) == 0:
            stats[f] = (0, 0)
            continue
        try:
            images = read_images(f)
        except Exception as exc:  # noqa: BLE001 - any unreadable image: skip the frame, say so below
            skipped.append((f, f"{type(exc).__name__}: {exc}"))
            stats[f] = (len(pos), len(pos))
            continue
        keep = light_keep(pos, ids, images, cals, cpar, z_min, project)
        stats[f] = (int(keep.sum()), len(pos))
        if keep.all():
            continue
        rcm = store.read_rcm(f)
        store.write_correspondences(f, pos[keep], ids[keep])
        if rcm is not None and len(rcm) == len(pos):
            store.write_rcm(f, rcm[keep])
    if skipped:
        warnings.warn(
            f"light check skipped {len(skipped)} of {len(list(frames))} frames because their images could not be read "
            f"(first: frame {skipped[0][0]}, {skipped[0][1]}); those frames keep all their points.",
            RuntimeWarning,
            stacklevel=2,
        )
    return stats


__all__ = [
    "DEFAULT_Z_MIN",
    "apply_light_check",
    "light_check_from_params",
    "light_keep",
    "light_z",
    "missing_view_light",
]
