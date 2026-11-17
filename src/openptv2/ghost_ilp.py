"""Spatial uniqueness via ILP (and Hungarian for bipartite components).

Given candidate 3D points per frame (with their 2D target ids), select the
subset that maximises total confidence subject to each 2D target being used by
at most one selected point. This is the multi-dimensional assignment problem;
``scipy.optimize.milp`` solves the general k-camera case, while
``linear_sum_assignment`` handles the 2-point (bipartite) case more cheaply.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds, linear_sum_assignment

from .ghost_components import conflict_components, classify_component


def _confidences(pos: np.ndarray, cam_ids: np.ndarray, conf: np.ndarray | None) -> np.ndarray:
    n = len(pos)
    if conf is None:
        return np.ones(n, dtype=float)
    conf = np.asarray(conf, dtype=float).ravel()
    if conf.shape[0] != n:
        raise ValueError(f"conf shape {conf.shape} != n={n}")
    return conf


def _targets_of_point(cam_ids: np.ndarray, i: int) -> list[tuple[int, int]]:
    """Return list of (camera, target_id) pairs used by point i."""
    out = []
    row = cam_ids[i]
    for cam, tid in enumerate(row):
        if tid is not None and int(tid) >= 0:
            out.append((cam, int(tid)))
    return out


def solve_bipartite_pair(
    pos: np.ndarray,
    cam_ids: np.ndarray,
    i: int,
    j: int,
    conf: np.ndarray,
) -> list[int]:
    """Decide which of two conflicting points to keep.

    If they share no target, keep both. Otherwise keep the higher-confidence
    one (ties broken by index).
    """
    ti = set(_targets_of_point(cam_ids, i))
    tj = set(_targets_of_point(cam_ids, j))
    if not (ti & tj):
        return [i, j]
    return [i] if conf[i] >= conf[j] else [j]


def solve_component_ilp(
    pos: np.ndarray,
    cam_ids: np.ndarray,
    comp: Sequence[int],
    conf: np.ndarray,
    max_size: int = 20,
    time_limit_s: float = 1.0,
) -> list[int]:
    """Select a subset of ``comp`` point indices maximising confidence s.t.
    each 2D target used at most once. Returns selected indices.

    Falls back to greedy (keep highest conf, drop conflicts) on overflow or
    solver failure.
    """
    comp = list(comp)
    m = len(comp)
    if m == 0:
        return []
    if m == 1:
        return [comp[0]]
    if m == 2:
        return solve_bipartite_pair(pos, cam_ids, comp[0], comp[1], conf)

    if m > max_size:
        # Greedy fallback.
        return _greedy_select(pos, cam_ids, comp, conf)

    # Build target list for this component.
    target_index: dict[tuple[int, int], int] = {}
    point_targets: list[list[int]] = []
    for idx in comp:
        pts = []
        for (cam, tid) in _targets_of_point(cam_ids, idx):
            key = (cam, tid)
            if key not in target_index:
                target_index[key] = len(target_index)
            pts.append(target_index[key])
        point_targets.append(pts)

    n_targets = len(target_index)
    # Objective: maximise sum conf_i x_i  -> minimise -conf.
    c = -np.asarray([conf[i] for i in comp], dtype=float)
    # Constraints: for each target t, sum_{i: t in point_i} x_i <= 1.
    A = np.zeros((n_targets, m), dtype=float)
    for k, pts in enumerate(point_targets):
        for t in pts:
            A[t, k] = 1.0
    constraints = LinearConstraint(A, -np.inf, np.ones(n_targets))
    integrality = np.ones(m)
    bounds = Bounds(0.0, 1.0)
    options = {"time_limit": time_limit_s, "disp": False}
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality,
                   bounds=bounds, options=options)
    except Exception:
        return _greedy_select(pos, cam_ids, comp, conf)
    if res.x is None:
        return _greedy_select(pos, cam_ids, comp, conf)
    x = res.x
    selected = [comp[k] for k in range(m) if x[k] > 0.5]
    if not selected:
        # Solver returned nothing; fall back to greedy to avoid emptying frame.
        return _greedy_select(pos, cam_ids, comp, conf)
    return selected


def _greedy_select(
    pos: np.ndarray,
    cam_ids: np.ndarray,
    comp: Sequence[int],
    conf: np.ndarray,
) -> list[int]:
    """Greedy: sort by conf desc, keep a point unless it conflicts with kept."""
    order = sorted(comp, key=lambda i: conf[i], reverse=True)
    kept: list[int] = []
    used_targets: set[tuple[int, int]] = set()
    for i in order:
        ts = set(_targets_of_point(cam_ids, i))
        if ts & used_targets:
            continue
        kept.append(i)
        used_targets |= ts
    if not kept and comp:
        # keep at least the best
        best = max(comp, key=lambda i: conf[i])
        kept.append(best)
    return kept


def ilp_uniqueness_frame(
    pos: np.ndarray,
    cam_ids: np.ndarray,
    conf: np.ndarray | None = None,
    max_size: int = 20,
    time_limit_s: float = 1.0,
) -> np.ndarray:
    """Return a boolean keep-mask (length len(pos)) selecting a unique subset.

    Components are solved independently; results combined.
    """
    pos = np.asarray(pos)
    cam_ids = np.asarray(cam_ids)
    n = len(pos)
    if n == 0:
        return np.zeros(0, dtype=bool)
    conf_arr = _confidences(pos, cam_ids, conf)
    comps = conflict_components(pos, cam_ids)
    keep = np.zeros(n, dtype=bool)
    for comp in comps:
        if len(comp) == 1:
            keep[comp[0]] = True
            continue
        kind = classify_component(len(comp))
        if kind == "bipartite":
            sel = solve_bipartite_pair(pos, cam_ids, comp[0], comp[1], conf_arr)
        else:
            sel = solve_component_ilp(pos, cam_ids, comp, conf_arr,
                                     max_size=max_size, time_limit_s=time_limit_s)
        for i in sel:
            keep[i] = True
    return keep


def resolve_ilp_config(track_cfg: dict, regime: dict) -> bool:
    """Resolve ilp_uniqueness auto/true/false from config + regime."""
    val = track_cfg.get("ilp_uniqueness", "auto")
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("auto", ""):
            val = "auto"
        elif v in ("1", "true", "yes", "on"):
            return True
        elif v in ("0", "false", "no", "off"):
            return False
    if val is True:
        return True
    if val is False:
        return False
    # auto: conservative — per-frame ILP can fragment tracks, so only enable
    # at very high ghost rates. Trajectory-level filtering is preferred.
    ghost_rate = float(regime.get("ghost_rate", 0.0) or 0.0)
    density = float(regime.get("density", 0.0) or 0.0)
    return ghost_rate > 0.4 or density >= 2000.0