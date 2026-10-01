"""Measure a run's data and say which two_phase tracking parameters suit it.

Every number the tracker's data-dependent rules look at is measured here from the
stored correspondences (and a short first tracking pass), and every recommendation is
printed with the measurement that led to it. See ``docs/tracking_parameters_guide.md``.

    measure(...)  -> dict of measurements
    advise(m)     -> list of Advice (parameter, value, reason)
    report(m, a)  -> text, ends with YAML lines to paste into the ``track:`` section

Rules (from the benchmark, docs/plans/2026-09-30-tracker-plan.md):

* ghost marks  (q_seed, q_young): many 3-camera points / many flagged points -> stronger
  rules; clean data -> lighter rules.
* confirmation tolerance (confirm_tol / confirm_auto): follows the position noise when the
  data are sparse and the kink is noise dominated, otherwise the fixed 0.3 mm/frame.
* blob_gate: for frame-skipped or low-frame-rate data (large steps).
* smoothing window: about 4 ms of frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Advice:
    parameter: str
    value: Any
    reason: str


def measure(
    store: Any,
    frames: list[int],
    cals: list | None = None,
    cpar: Any = None,
    v_max: float = 1.0,
    n_track: int = 60,
    radius: float = 5.0,
) -> dict[str, float]:
    """Measurements the data-dependent rules look at.

    ``cals``/``cpar`` (camera models) are needed for the ghost-mark numbers; without
    them those entries are NaN. A short first tracking pass (no confirmation, no ghost
    rules, 3D costs) over ``n_track`` frames gives the step and the kink."""
    from scipy.spatial import cKDTree

    from openptv2.plugins.two_phase_tracking import (
        TwoPhaseTracker,
        TwoPhaseTrackerConfig,
    )
    from openptv2.point_quality import (
        brightness_spread,
        fit_scale,
        ghost_probability,
        log_brightness_from_arrays,
        rcm_from_arrays,
        read_frame_arrays,
    )
    from openptv2.storage.run_store import seen_mask
    from openptv2.tracking_postprocess import kink_statistics

    m: dict[str, float] = {"frames": float(len(frames))}
    pos_l, nseen_l, rcm_l, logb_l = [], [], [], []
    nb = []
    for i, f in enumerate(frames):
        pos, cam_ids, rows = read_frame_arrays(store, f)
        pos_l.append(np.asarray(pos))
        nseen_l.append(seen_mask(cam_ids).sum(axis=1))
        if cals and cpar is not None and i % max(1, len(frames) // 40) == 0:
            rcm_l.append(rcm_from_arrays(pos, cam_ids, rows, cals, cpar))
            logb_l.append(log_brightness_from_arrays(cam_ids, rows))
        else:
            rcm_l.append(None)
            logb_l.append(None)
        if i % max(1, len(frames) // 6) == 0 and len(pos) > 10:
            tree = cKDTree(pos)
            nb.append(
                np.median([len(x) - 1 for x in tree.query_ball_point(pos[::6], radius)])
            )
    m["points_per_frame"] = float(np.median([len(p) for p in pos_l]))
    m["neighbours"] = float(np.mean(nb)) if nb else float("nan")
    allseen = np.concatenate(nseen_l)
    m["three_cam_share"] = (
        float(np.mean(allseen == 3)) if len(allseen) else float("nan")
    )

    # ghost marks: on the frames where rcm was computed
    idx = [i for i, r in enumerate(rcm_l) if r is not None]
    if idx:
        P = np.vstack([pos_l[i] for i in idx])
        R = np.concatenate([rcm_l[i] for i in idx])
        N = np.concatenate([nseen_l[i] for i in idx])
        scale = fit_scale(P, R, N)
        allb = np.vstack([logb_l[i] for i in idx])
        with np.errstate(all="ignore"):
            off = np.nanmedian(allb, axis=0)
            varies = float(np.nanstd(allb)) > 1e-6
        g = np.concatenate(
            [
                ghost_probability(
                    pos_l[i],
                    rcm_l[i],
                    nseen_l[i],
                    scale,
                    brightness_spread(logb_l[i], off) if varies else None,
                )
                for i in idx
            ]
        )
        m["flagged_02"] = float(np.mean(g > 0.2))
        m["flagged_05"] = float(np.mean(g > 0.5))
    else:
        m["flagged_02"] = m["flagged_05"] = float("nan")

    # short first tracking pass for step and kink
    sub = pos_l[: max(3, min(n_track, len(pos_l)))]
    cfg = TwoPhaseTrackerConfig(
        v_max=v_max,
        cost_mode="3d",
        leaf_weight=0.0,
        max_gap=1,
        confirm_tol=None,
        confirm_ends=False,
        q_seed=None,
        q_young=0,
    )
    links = TwoPhaseTracker(cfg).track_frames(sub, None)
    st = kink_statistics(links, sub, radius=radius)
    m["median_step"] = st["median_step"]
    m["median_kink"] = st["median_kink"]
    m["kink_step_ratio"] = st["ratio"]
    return m


def advise(
    m: dict[str, float],
    fps: float | None = None,
    max_neighbours: float = 7.0,
    min_ratio: float = 0.5,
    multiplier: float = 8.0,
) -> list[Advice]:
    """Recommendations from the measurements (see the module docstring)."""
    out: list[Advice] = []
    c3, f02 = m.get("three_cam_share", np.nan), m.get("flagged_02", np.nan)

    # 1. ghost rules
    if np.isfinite(f02) and (c3 > 0.45 or f02 > 0.14):
        out += [
            Advice(
                "q_seed",
                0.15,
                f"{c3:.0%} of the points have 3 cameras and {f02:.0%} are flagged "
                "(typical: 35-40% and about 10%): many doubtful points, so the "
                "accuracy mode removes more ghosts",
            ),
            Advice("q_young", 6, "accuracy mode (together with q_seed 0.15)"),
        ]
    elif np.isfinite(f02) and c3 < 0.20 and f02 < 0.04:
        out += [
            Advice(
                "q_seed",
                0.3,
                f"only {c3:.0%} 3-camera points and {f02:.0%} flagged: few ghosts, "
                "a lighter rule keeps more points",
            ),
            Advice("q_young", 3, "default"),
        ]
    else:
        out += [
            Advice(
                "q_seed",
                0.2,
                "default: "
                + (
                    f"{c3:.0%} 3-camera points, {f02:.0%} flagged"
                    if np.isfinite(f02)
                    else "no camera models given, ghost share not measured"
                ),
            ),
            Advice("q_young", 3, "default"),
        ]

    # 2. confirmation tolerance
    nb, ratio, kink, step = (
        m["neighbours"],
        m["kink_step_ratio"],
        m["median_kink"],
        m["median_step"],
    )
    sparse = nb <= max_neighbours
    noisy = ratio >= min_ratio
    if sparse and noisy:
        tol = multiplier * kink
        out.append(
            Advice(
                "confirm_tol",
                "leave unset (automatic)",
                f"sparse ({nb:.1f} neighbours within 5 mm, limit {max_neighbours:g}) and "
                f"noise dominated (median kink {kink:.3f} = {ratio:.2f} x median step "
                f"{step:.3f}, limit {min_ratio:g}): the tolerance is set from the data, "
                f"about {tol:.2f} mm/frame. A pinned `confirm_tol:` in the yaml would "
                "switch this off.",
            )
        )
    else:
        why = []
        if not sparse:
            why.append(
                f"dense ({nb:.1f} neighbours within 5 mm, limit {max_neighbours:g})"
            )
        if not noisy:
            why.append(
                f"motion dominates the kink (median kink {kink:.3f} = {ratio:.2f} x "
                f"median step {step:.3f}, limit {min_ratio:g}; frame skipping?)"
            )
        out.append(
            Advice(
                "confirm_tol",
                0.3,
                "the automatic tolerance is NOT safe here: "
                + " and ".join(why)
                + "; use the fixed 0.3 mm/frame (or tighten for dense data)",
            )
        )

    # 3. brightness continuity gate
    if np.isfinite(step) and step >= 0.2:
        out.append(
            Advice(
                "blob_gate",
                0.5,
                f"large steps ({step:.2f} mm/frame: frame-skipped or low frame rate): "
                "brightness continuity removes wrong candidates",
            )
        )
    # 4. smoothing window
    if fps:
        w = max(5, int(round(0.004 * fps)) // 2 * 2 + 1)
        out.append(
            Advice(
                "trajectories.smoothing_window",
                w,
                f"about 4 ms of frames at {fps:g} fps (longer = smoother velocities but "
                "blurs real accelerations; the benchmark optimum was 21 at 5000 fps)",
            )
        )
    return out


def report(m: dict[str, float], advice: list[Advice]) -> str:
    """Readable report: measurements, then each recommendation with its reason."""
    lines = ["MEASURED ON YOUR DATA", "---------------------"]
    names = [
        ("points_per_frame", "3D points per frame (median)", "{:.0f}"),
        (
            "neighbours",
            "other points within 5 mm of a point (median) = density",
            "{:.1f}",
        ),
        ("three_cam_share", "share of points seen by only 3 cameras", "{:.0%}"),
        ("flagged_02", "share of points with ghost probability > 0.2", "{:.1%}"),
        ("flagged_05", "share of points with ghost probability > 0.5", "{:.1%}"),
        ("median_step", "median step per frame (mm)", "{:.3f}"),
        (
            "median_kink",
            "median kink = change of the step (mm), measures the noise",
            "{:.3f}",
        ),
        ("kink_step_ratio", "kink / step (high = noise dominated)", "{:.2f}"),
    ]
    for key, label, fmt in names:
        v = m.get(key, float("nan"))
        lines.append(f"  {label:62s} " + (fmt.format(v) if np.isfinite(v) else "n/a"))
    lines += ["", "RECOMMENDATION", "--------------"]
    for a in advice:
        lines.append(f"  {a.parameter} = {a.value}")
        lines.append(f"      why: {a.reason}")
    lines += ["", "YAML (track: section)", "---------------------", "track:"]
    for a in advice:
        if a.parameter.startswith("trajectories."):
            continue
        if isinstance(a.value, str):
            lines.append(f"  # {a.parameter}: {a.value}")
        else:
            lines.append(f"  {a.parameter}: {a.value}")
    tw = [a for a in advice if a.parameter.startswith("trajectories.")]
    if tw:
        lines += ["trajectories:", f"  smoothing_window: {tw[0].value}"]
    return "\n".join(lines)
