"""Pure, read-only auto-parameter estimator for the unified tracker (Step 1b).

Derives motion, density, conditioning, and ghost-rate estimates from a
``RunStore`` (sampled across ``[first, last]`` via ``sample_probe_windows``)
and camera calibration (``z_noise_floor_mm``), without mutating the
experiment or user configuration.
"""

from __future__ import annotations

import statistics
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
from scipy.spatial import cKDTree

from openptv2.plugins.unified_tracking import Tracking
from openptv2.tracking_feasibility import measure_motion_scale, z_noise_floor_mm


def _is_auto(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, str):
        return val.strip().lower() in ("auto", "")
    return False


def _parse_bool(val: Any) -> bool:
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "on")
    return bool(val)


def _unwrap_exp(exp: Any) -> tuple[Any, Any, Any, Any, Any]:
    """Extract (pm, spar, store, cals, cpar) from an experiment or wrapper."""
    if exp is None:
        return None, None, None, None, None

    inner = getattr(exp, "exp1", None)
    pm = getattr(exp, "pm", None) or (getattr(inner, "pm", None) if inner else None)
    spar = getattr(exp, "spar", None) or (getattr(inner, "spar", None) if inner else None)
    store = (
        getattr(exp, "store", None)
        or getattr(getattr(exp, "tracker", None), "_store", None)
        or (getattr(inner, "store", None) if inner else None)
    )
    cals = getattr(exp, "cals", None) or (getattr(inner, "cals", None) if inner else None)
    cpar = getattr(exp, "cpar", None) or (getattr(inner, "cpar", None) if inner else None)
    return pm, spar, store, cals, cpar


def _resolve_frame_range(
    spar: Any, pm: Any, store: Any
) -> tuple[Optional[int], Optional[int]]:
    first: Optional[int] = None
    last: Optional[int] = None
    if spar is not None:
        if hasattr(spar, "get_first"):
            first = spar.get_first()
        elif hasattr(spar, "first"):
            first = getattr(spar, "first")
        if hasattr(spar, "get_last"):
            last = spar.get_last()
        elif hasattr(spar, "last"):
            last = getattr(spar, "last")

    if (first is None or last is None) and pm is not None:
        seq = (getattr(pm, "parameters", None) or {}).get("sequence", {}) or {}
        if first is None and "first" in seq and seq["first"] is not None:
            first = int(seq["first"])
        if last is None and "last" in seq and seq["last"] is not None:
            last = int(seq["last"])

    if (first is None or last is None) and store is not None and hasattr(store, "frames"):
        frames = store.frames("correspondences")
        if frames:
            if first is None:
                first = int(frames[0])
            if last is None:
                last = int(frames[-1])

    return (
        int(first) if first is not None else None,
        int(last) if last is not None else None,
    )


def _read_finite_positions(store: Any, frame: int) -> Optional[np.ndarray]:
    if not store.has_correspondences(frame):
        return None
    pos, _ = store.read_correspondences(frame)
    arr = np.asarray(pos, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.empty((0, 3), dtype=float)
    mask = np.isfinite(arr).all(axis=1)
    return arr[mask]


def make_synthetic_exp(
    store: Any,
    first: int,
    last: int,
    *,
    cals: Any = None,
    cpar: Any = None,
    track_cfg: Optional[dict[str, Any]] = None,
) -> SimpleNamespace:
    """Build a lightweight mock experiment object for ``estimate_params``.

    Intended for tests and offline estimation against a ``RunStore`` + optional
    calibration objects without constructing a full GUI/processing experiment.
    """
    pm = SimpleNamespace(
        parameters={
            "sequence": {"first": int(first), "last": int(last)},
            "track": dict(track_cfg or {}),
        }
    )
    spar = SimpleNamespace(
        get_first=lambda: int(first),
        get_last=lambda: int(last),
    )
    return SimpleNamespace(
        pm=pm,
        spar=spar,
        store=store,
        cals=cals,
        cpar=cpar,
    )


def estimate_params(
    exp: Any,
    track_cfg: Optional[dict[str, Any]] = None,
    *,
    windows: Optional[list[tuple[int, int]]] = None,
) -> dict[str, Any]:
    """Estimate tracking parameters from correspondences and calibration.

    Pure and read-only: never mutates ``exp``, ``store``, or ``track_cfg``.

    Parameters
    ----------
    exp : Any
        Experiment object (or ``SimpleNamespace`` from ``make_synthetic_exp``).
    track_cfg : dict, optional
        Tracking configuration overrides. If omitted, read from ``exp.pm``.
    windows : list[tuple[int, int]], optional
        Explicit ``(start, end)`` frame windows to sample. Defaults to
        ``Tracking.sample_probe_windows(first, last)``.

    Returns
    -------
    dict[str, Any]
        Resolved parameter estimates merged with explicit user overrides.
    """
    pm, spar, store, cals, cpar = _unwrap_exp(exp)
    if track_cfg is None:
        cfg = dict((getattr(pm, "parameters", None) or {}).get("track", {}) or {})
    else:
        cfg = dict(track_cfg)

    v_max_safety = float(cfg.get("v_max_safety", 1.5) or 1.5)
    detection_noise_px = float(cfg.get("detection_noise_px", 0.5) or 0.5)

    first, last = _resolve_frame_range(spar, pm, store)
    if windows is None:
        if first is not None and last is not None and last >= first:
            sampled_windows = Tracking.sample_probe_windows(first, last)
        else:
            sampled_windows = []
    else:
        sampled_windows = list(windows)

    counts_per_frame: list[int] = []
    window_displacements: list[float] = []
    all_pair_displacements: list[float] = []
    all_ipds: list[float] = []
    all_p95_disps: list[float] = []
    pair_positions: list[tuple[np.ndarray, np.ndarray]] = []
    rcm_ghost_probs: list[float] = []

    if store is not None:
        for w_start, w_end in sampled_windows:
            frame_pos: dict[int, np.ndarray] = {}
            win_disps: list[float] = []
            for fn in range(int(w_start), int(w_end) + 1):
                pos = _read_finite_positions(store, fn)
                if pos is None:
                    continue
                counts_per_frame.append(int(len(pos)))
                frame_pos[fn] = pos
                if hasattr(store, "read_point_quality"):
                    pq = store.read_point_quality(fn)
                    if pq is not None and len(pq) > 0:
                        rcm_ghost_probs.append(float(np.mean(np.asarray(pq, dtype=float))))

            sorted_frames = sorted(frame_pos.keys())
            for f_a, f_b in zip(sorted_frames[:-1], sorted_frames[1:]):
                if f_b != f_a + 1:
                    continue
                pos_a = frame_pos[f_a]
                pos_b = frame_pos[f_b]
                if len(pos_a) > 0 and len(pos_b) > 0:
                    pair_positions.append((pos_a, pos_b))

                scale = measure_motion_scale(pos_a, pos_b)
                if scale is not None:
                    disp, _spacing = scale
                    win_disps.append(disp)
                    all_pair_displacements.append(disp)

                if len(pos_a) >= 2 and len(pos_b) >= 1:
                    tree_a = cKDTree(pos_a)
                    d_aa, _ = tree_a.query(pos_a, k=2)
                    mean_ipd = float(np.mean(d_aa[:, 1]))
                    if mean_ipd > 0:
                        all_ipds.append(mean_ipd)

                    tree_b = cKDTree(pos_b)
                    d_ab, _ = tree_b.query(pos_a, k=1)
                    all_p95_disps.append(float(np.percentile(d_ab, 95)))

            if win_disps:
                window_displacements.append(float(statistics.median(win_disps)))

    density = float(statistics.median(counts_per_frame)) if counts_per_frame else 0.0
    density_max = int(max(counts_per_frame)) if counts_per_frame else 0

    displacement = (
        float(statistics.median(all_pair_displacements))
        if all_pair_displacements
        else None
    )
    mean_ipd = float(statistics.median(all_ipds)) if all_ipds else None
    p95_disp = float(statistics.median(all_p95_disps)) if all_p95_disps else None

    rho: Optional[float] = None
    if p95_disp is not None and mean_ipd is not None and mean_ipd > 0:
        rho = p95_disp / mean_ipd

    v_max: Optional[float] = None
    if window_displacements:
        v_max = max(window_displacements) * v_max_safety
        if rho is not None and rho > 1.0 and mean_ipd is not None and mean_ipd > 0:
            v_max = min(v_max, mean_ipd)

    noise_floor_z: Optional[float] = None
    if cals and cpar is not None:
        z_val = float(
            z_noise_floor_mm(cals, cpar, detection_noise_px=detection_noise_px)
        )
        noise_floor_z = z_val

    conditioning_ratio: Optional[float] = None
    if (
        displacement is not None
        and noise_floor_z is not None
        and noise_floor_z > 0.0
    ):
        conditioning_ratio = displacement / noise_floor_z

    slow_flow = bool(conditioning_ratio is not None and conditioning_ratio < 1.0)

    # Merge explicit user overrides (user wins whenever not "auto" / None).
    if not _is_auto(cfg.get("v_max")):
        v_max = float(cfg["v_max"])
    if not _is_auto(cfg.get("slow_flow")):
        slow_flow = _parse_bool(cfg["slow_flow"])
    if not _is_auto(cfg.get("density")):
        density = float(cfg["density"])
    elif not _is_auto(cfg.get("density_estimate")):
        d_est = float(cfg["density_estimate"])
        if d_est > 0:
            density = d_est

    # Ghost rate: fraction of correspondences with no plausible successor within v_max.
    ghost_rate: Optional[float] = None
    if v_max is not None and v_max > 0 and pair_positions:
        total_pts = 0
        unmatched_pts = 0
        for pos_a, pos_b in pair_positions:
            tree_b = cKDTree(pos_b)
            nearest_d, _ = tree_b.query(pos_a, k=1)
            total_pts += len(pos_a)
            unmatched_pts += int(np.sum(nearest_d > v_max))
        if total_pts > 0:
            ghost_rate = unmatched_pts / total_pts
    elif rcm_ghost_probs:
        ghost_rate = float(statistics.median(rcm_ghost_probs))

    return {
        "v_max": v_max,
        "density": density,
        "density_max": density_max,
        "noise_floor_z": noise_floor_z,
        "conditioning_ratio": conditioning_ratio,
        "slow_flow": slow_flow,
        "ghost_rate": ghost_rate,
        "displacement": displacement,
        "mean_interparticle_distance": mean_ipd,
        "rho": rho,
        "windows_sampled": sampled_windows,
    }
