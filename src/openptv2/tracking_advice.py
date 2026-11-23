"""Which tracker, and which options, for this data: measured on the run's own points, no truth needed.

The rules come from a sweep of 13 benchmark cases (noise, merge rate, density, frame skip, clustering) and two real
recordings; the table and the reasons are in ``docs/trackers.md`` ("Choosing"). Six numbers describe the data
(:func:`describe_store`); :func:`advise` turns them into a tracker and the optional extras, with the reason for each.

The extras are never on by default. They are for difficult data (overlapping particles), where they bring a
solid but not drastic improvement at a cost in run time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: ``kink_over_step`` below this means exact (simulated) data: two_phase's ghost rules then only cost points.
EXACT_KINK_OVER_STEP = 0.1
#: share of 3-4 camera points with a blob much brighter than their other views (the ``shared_pass`` test)
CORRECTION_RECOMMENDED = 0.10      # velocity error -13 % (flag 0.11, with the pass) to -40 %; real recordings: 0.15
CORRECTION_OPTIONAL = 0.08         # below 0.10 the gain is a few percent: not worth the run time unless every bit counts
PASS_MIN_FLAG_RATE = 0.10
#: share of points with a mutual nearest-neighbour link; below it the data is dense and the pass alone adds noise
SPARSE_MUTUAL_RATE = 0.9
#: how many neighbour spacings fit in one step; below it large steps break the 3D-search trackers
LARGE_STEP_SPACING_OVER_STEP = 10.0
BRIGHTER_RATIO = 1.7


@dataclass
class RunDescription:
    n_frames: int
    ppf: float                       # points per frame
    nn_mm: float                     # median nearest-neighbour spacing in a frame
    step_mm: float                   # median mutual-nearest-neighbour step between frames
    spacing_over_step: float
    mutual_rate: float
    kink_over_step: float            # noise against motion
    frac_partial: float              # share of points seen in fewer cameras than the rig has
    merge_flag_rate: float           # proxy for the share of blobs that hold two particles

    def as_dict(self) -> dict[str, float]:
        return {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


@dataclass
class Advice:
    tracker: str = "two_phase"
    shared_pass: bool = False
    shared_correction: bool = False
    reasons: list[str] = field(default_factory=list)
    description: RunDescription | None = None

    def settings(self) -> dict[str, Any]:
        """The YAML keys: ``track`` (tracker, shared_correction) and ``ptv`` (shared_pass)."""
        return {"track": {"tracker": self.tracker, "shared_correction": self.shared_correction},
                "ptv": {"shared_pass": self.shared_pass}}


def describe_store(store: Any, first: int | None = None, last: int | None = None, n_sample: int = 30,
                   ratio: float = BRIGHTER_RATIO) -> RunDescription:
    """Measure the six numbers on ``n_sample`` frames of a run store (targets and correspondences)."""
    from scipy.spatial import cKDTree

    frames = [f for f in store.frames() if (first is None or f >= first) and (last is None or f <= last)]
    if not frames:
        raise ValueError("the store has no frames in that range")
    n_cams = len([k for k in store.root["targets"].keys()]) if "targets" in store.root else 4
    pick = frames[:: max(1, len(frames) // n_sample)][:n_sample]
    have = set(frames)
    pts, ids, tg = {}, {}, {}
    for f in sorted({g for f in pick for g in (f, f + 1, f + 2) if g in have}):
        p, i = store.read_correspondences(f)
        pts[f], ids[f] = np.asarray(p, float), np.asarray(i)[:, :n_cams]
        tg[f] = [np.asarray(store.root[f"targets/cam_{c}/frame_{f:06d}"]) for c in range(n_cams)]
    nn, rows, partial, flagged, judged = [], 0, 0, 0, 0
    for f in pick:
        p = pts[f]
        if len(p) < 3:
            continue
        nn.extend(cKDTree(p).query(p, k=2)[0][:, 1])
        rows += len(p)
        partial += int(((ids[f] >= 0).sum(1) < n_cams).sum())
        gain = [float(np.median(t[:, 6])) if len(t) else 1.0 for t in tg[f]]
        b = np.full((len(p), n_cams), np.nan)
        for c in range(n_cams):
            m = ids[f][:, c] >= 0
            b[m, c] = tg[f][c][ids[f][m, c], 6] / max(gain[c], 1e-9)
        ok = (~np.isnan(b)).sum(1) >= 3
        if ok.any():
            ref = np.sort(np.where(np.isnan(b[ok]), np.inf, b[ok]), axis=1)[:, 1]
            flagged += int((np.nan_to_num(b[ok], nan=0) > ratio * ref[:, None]).any(1).sum())
            judged += int(ok.sum())
    nn_med = float(np.median(nn)) if nn else float("nan")
    steps, mutual, chains, kinks = [], [], {}, []
    for f in sorted(pts):
        if f + 1 not in pts or len(pts[f]) < 3 or len(pts[f + 1]) < 3:
            continue
        a, b2 = pts[f], pts[f + 1]
        da, ia = cKDTree(b2).query(a)
        _db, ib = cKDTree(a).query(b2)
        good = (ib[ia] == np.arange(len(a))) & (da < 0.5 * nn_med)
        mutual.append(good.mean())
        steps.extend(da[good])
        chains[f] = (np.flatnonzero(good), ia[good])
    step = float(np.median(steps)) if steps else float("nan")
    for f in sorted(pts):
        if f not in chains or f + 1 not in chains or f + 2 not in pts:
            continue
        nxt = dict(zip(*chains[f + 1]))
        for i, j in zip(*chains[f]):
            k = nxt.get(int(j))
            if k is not None:
                kinks.append(np.linalg.norm(pts[f][i] - 2 * pts[f + 1][j] + pts[f + 2][k]))
    kink = float(np.median(kinks)) if kinks else float("nan")
    ok_step = step == step and step > 0
    return RunDescription(
        n_frames=len(frames), ppf=rows / max(len(pick), 1), nn_mm=nn_med, step_mm=step,
        spacing_over_step=nn_med / step if ok_step else float("nan"),
        mutual_rate=float(np.mean(mutual)) if mutual else float("nan"),
        kink_over_step=kink / step if ok_step and kink == kink else float("nan"),
        frac_partial=partial / max(rows, 1), merge_flag_rate=flagged / max(judged, 1),
    )


def advise(d: RunDescription) -> Advice:
    """Tracker and optional extras for data described by ``d``."""
    a = Advice(description=d)
    exact = d.kink_over_step == d.kink_over_step and d.kink_over_step < EXACT_KINK_OVER_STEP
    if exact:
        a.tracker = "fast_3d"
        a.reasons.append(
            f"Exact data (noise {d.kink_over_step:.2f} of a step, no merged blobs): fast_3d keeps every point "
            "(yield 0.995 on the benchmark) where two_phase's ghost rules drop 3 to 6 % of them. trackcorr gives the "
            "same result about 2 times slower; use it to reproduce legacy results.")
        return a
    a.reasons.append(
        "two_phase: the fastest and the most accurate on noisy data (benchmark: 1.3 to 7 times lower velocity error "
        "than trackcorr or fast_3d, 4 to 6 times faster than trackcorr). It is the only one that copes with large "
        "steps (neighbours " + (f"{d.spacing_over_step:.0f}" if d.spacing_over_step == d.spacing_over_step else "?") +
        " steps apart here; below 10 the others fail).")
    flag = d.merge_flag_rate
    if flag >= CORRECTION_RECOMMENDED:
        a.shared_correction = True
        a.reasons.append(
            f"shared_correction recommended: {flag:.0%} of the points have a blob that looks shared with another "
            "particle (benchmark: -13 % to -40 % velocity error above 10 %; real recordings with 15 %: tracks of "
            "200+ frames doubled, 15 % smoother). Costs about 3 to 4 times the tracking time.")
    elif flag >= CORRECTION_OPTIONAL:
        a.reasons.append(
            f"shared_correction optional: {flag:.0%} look shared (a few percent lower velocity error); "
            "usually not worth 3 to 4 times the tracking time.")
    else:
        a.reasons.append(f"No extras: only {flag:.0%} of the points have a shared blob (below 8 % they change nothing).")
    if flag >= PASS_MIN_FLAG_RATE:
        sparse = d.mutual_rate == d.mutual_rate and d.mutual_rate >= SPARSE_MUTUAL_RATE
        if sparse or a.shared_correction:
            a.shared_pass = True
            a.reasons.append(
                "ptv.shared_pass recommended for more points and longer tracks (+2 points of yield, +3 to 4 points of "
                "long-track completeness); its new points are noisy, so it works best together with shared_correction. "
                "It adds about 10 to 30 % to the matching time.")
        else:
            a.reasons.append(
                f"ptv.shared_pass not advised on its own here: the data is dense (mutual-link share "
                f"{d.mutual_rate:.2f}) and its noisy new points raise the velocity error (+3 %); with "
                "shared_correction it is fine.")
    return a


def format_advice(a: Advice) -> str:
    d = a.description
    lines = ["  Setup advice, measured on your points (docs/trackers.md, \"Choosing\"):"]
    if d is not None:
        lines.append(
            f"    {d.ppf:.0f} points/frame, spacing/step {d.spacing_over_step:.1f}, mutual links {d.mutual_rate:.2f}, "
            f"noise/step {d.kink_over_step:.2f}, partial views {d.frac_partial:.2f}, shared-blob flag {d.merge_flag_rate:.2f}")
    lines.append(f"    tracker: {a.tracker}")
    lines.append(f"    track.shared_correction: {str(a.shared_correction).lower()}   ptv.shared_pass: {str(a.shared_pass).lower()}")
    for r in a.reasons:
        lines.append(f"    - {r}")
    return "\n".join(lines)


__all__ = ["Advice", "RunDescription", "advise", "describe_store", "format_advice"]
