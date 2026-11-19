"""Minimal Synthetic Ground-Truth Scorecard (H0).

Fast (<15-20s), deterministic 3-regime synthetic GT benchmark used to calibrate
and regression-test unified tracker thresholds across Waves 1-3:

1. ``clean``: moderate density, clean motion
2. ``slow_noisy``: slow flow where frame-to-frame displacement is small relative
   to detection noise
3. ``dense_ghost``: higher density with ghost/ambiguous correspondences

Reports link-level ``yield``, ``precision``, ``median_track_length``, and
``jump_rate`` (large velocity-kink transitions), plus ``IdentityMetrics``
completeness/purity for reference.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import openptv2.benchmarking as bm
from openptv2.benchmarking.metrics import _frames_dict, _match_frame


@dataclass(frozen=True)
class RegimeSpec:
    name: str
    scenario: bm.ScenarioSpec
    jump_threshold_mm: float


REGIMES: tuple[RegimeSpec, ...] = (
    RegimeSpec(
        name="clean",
        scenario=bm.ScenarioSpec(
            num_particles=20,
            num_frames=10,
            velocity=1.5,
            velocity_jitter=0.2,
            noise_mm=0.0,
            ghost_ratio=0.0,
            seed=10,
        ),
        jump_threshold_mm=2.5,
    ),
    RegimeSpec(
        name="slow_noisy",
        scenario=bm.ScenarioSpec(
            num_particles=50,
            num_frames=20,
            velocity=0.5,
            velocity_jitter=0.05,
            noise_mm=0.15,
            ghost_ratio=0.0,
            seed=11,
            flow_type="turbulent",
        ),
        jump_threshold_mm=2.5,
    ),
    RegimeSpec(
        name="dense_ghost",
        scenario=bm.ScenarioSpec(
            num_particles=40,
            num_frames=10,
            velocity=1.2,
            velocity_jitter=0.3,
            noise_mm=0.1,
            ghost_ratio=0.35,
            seed=12,
        ),
        jump_threshold_mm=2.5,
    ),
)

TRACKERS: tuple[str, ...] = ("unified", "trackcorr", "priority_segment_3d")


@dataclass(frozen=True)
class ScorecardRow:
    regime: str
    tracker: str
    link_yield: float
    precision: float
    median_track_length: float
    jump_rate: float
    completeness: float
    purity: float
    elapsed_s: float


def _track_overrides(spec: bm.ScenarioSpec) -> dict[str, float]:
    dv = max(3.0, spec.velocity * 3.0 + 3.0 * spec.noise_mm)
    return dict(
        dvxmin=-dv,
        dvxmax=dv,
        dvymin=-dv,
        dvymax=dv,
        dvzmin=-dv,
        dvzmax=dv,
        dacc=max(3.0, dv),
    )


def compute_scorecard_metrics(
    true_tracks: dict[int, list[tuple[int, float, float, float]]],
    pred_tracks: dict[int, list[tuple[int, float, float, float]]],
    eps: float,
    jump_threshold_mm: float,
) -> tuple[float, float, float, float]:
    """Compute (link_yield, precision, median_track_length, jump_rate)."""
    true_by_frame = _frames_dict(true_tracks)
    pred_by_frame = _frames_dict(pred_tracks)

    all_frames = sorted(set(true_by_frame) | set(pred_by_frame))
    frame_matches: dict[int, dict[int, int]] = {
        f: _match_frame(true_by_frame.get(f, {}), pred_by_frame.get(f, {}), eps)
        for f in all_frames
    }

    total_true_links = 0
    for pts in true_tracks.values():
        pts_sorted = sorted(pts, key=lambda p: p[0])
        for i in range(len(pts_sorted) - 1):
            if pts_sorted[i + 1][0] == pts_sorted[i][0] + 1:
                total_true_links += 1

    total_pred_links = 0
    matched_true_links = 0
    n_transitions = 0
    n_jumps = 0

    for pred_id, pts in pred_tracks.items():
        pts_sorted = sorted(pts, key=lambda p: p[0])
        velocities: list[np.ndarray] = []
        for i in range(len(pts_sorted) - 1):
            f0, x0, y0, z0 = pts_sorted[i]
            f1, x1, y1, z1 = pts_sorted[i + 1]
            dt = max(1, f1 - f0)
            velocities.append(np.array([x1 - x0, y1 - y0, z1 - z0]) / dt)
            total_pred_links += 1
            if f1 == f0 + 1:
                t0 = frame_matches.get(f0, {}).get(pred_id)
                t1 = frame_matches.get(f1, {}).get(pred_id)
                if t0 is not None and t0 == t1:
                    matched_true_links += 1

        for i in range(len(velocities) - 1):
            n_transitions += 1
            if float(np.linalg.norm(velocities[i + 1] - velocities[i])) > jump_threshold_mm:
                n_jumps += 1

    link_yield = matched_true_links / total_true_links if total_true_links > 0 else 0.0
    precision = matched_true_links / total_pred_links if total_pred_links > 0 else 0.0
    lengths = [len(pts) for pts in pred_tracks.values()]
    median_len = float(np.median(lengths)) if lengths else 0.0
    jump_rate = n_jumps / n_transitions if n_transitions > 0 else 0.0
    return link_yield, precision, median_len, jump_rate


def _evaluate_cell(
    work_dir: Path, regime: RegimeSpec, tracker: str, first_frame: int = 10001
) -> ScorecardRow:
    true_tracks, frame_gt = bm.generate_scenario(regime.scenario)
    rig = bm.make_standard_rig(refract=False)
    yaml_path = bm.write_experiment(rig, frame_gt, work_dir, first_frame=first_frame)

    t0 = time.perf_counter()
    pred = bm.run_tracker(
        yaml_path, tracker, track_overrides=_track_overrides(regime.scenario)
    )
    elapsed_s = time.perf_counter() - t0

    true_tracks_abs = {
        pid: [(f + first_frame, x, y, z) for f, x, y, z in pts]
        for pid, pts in true_tracks.items()
    }
    eps = max(0.5, 2.0 * regime.scenario.noise_mm)
    link_yield, precision, median_len, jump_rate = compute_scorecard_metrics(
        true_tracks_abs, pred, eps=eps, jump_threshold_mm=regime.jump_threshold_mm
    )
    id_metrics = bm.compute_identity_metrics(true_tracks_abs, pred, eps=eps)
    return ScorecardRow(
        regime=regime.name,
        tracker=tracker,
        link_yield=link_yield,
        precision=precision,
        median_track_length=median_len,
        jump_rate=jump_rate,
        completeness=id_metrics.completeness,
        purity=id_metrics.purity,
        elapsed_s=elapsed_s,
    )


def _format_scorecard_table(rows: list[ScorecardRow]) -> str:
    header = (
        f"{'Regime':<12} {'Tracker':<20} {'Yield':>7} {'Prec':>7} "
        f"{'MedLen':>7} {'JumpRt':>7} {'Compl':>7} {'Purity':>7} {'Time(s)':>8}"
    )
    sep = "-" * len(header)
    lines = ["", "=== Unified Tracker Synthetic GT Scorecard (H0) ===", header, sep]
    for r in rows:
        lines.append(
            f"{r.regime:<12} {r.tracker:<20} {r.link_yield:7.3f} {r.precision:7.3f} "
            f"{r.median_track_length:7.1f} {r.jump_rate:7.3f} {r.completeness:7.3f} "
            f"{r.purity:7.3f} {r.elapsed_s:8.2f}"
        )
    lines.append(sep)
    return "\n".join(lines)


def test_unified_gt_scorecard(tmp_path: Path) -> None:
    """Run all 3 GT regimes across baseline trackers and print scorecard."""
    t_start = time.perf_counter()
    rows: list[ScorecardRow] = []

    for regime in REGIMES:
        for tracker in TRACKERS:
            cell_dir = tmp_path / f"{regime.name}_{tracker}"
            row = _evaluate_cell(cell_dir, regime, tracker)
            rows.append(row)

    total_s = time.perf_counter() - t_start
    print(_format_scorecard_table(rows))
    print(f"Total scorecard runtime: {total_s:.2f}s")

    assert total_s < 20.0, f"Scorecard exceeded 20s budget: {total_s:.2f}s"

    by_key = {(r.regime, r.tracker): r for r in rows}

    # Sanity floors so regressions are caught immediately
    for tracker in TRACKERS:
        clean = by_key[("clean", tracker)]
        assert clean.link_yield >= 0.70, f"{tracker} clean yield too low: {clean}"
        assert clean.precision >= 0.80, f"{tracker} clean precision too low: {clean}"
        assert clean.median_track_length >= 5.0, f"{tracker} clean med len too low: {clean}"
        assert clean.jump_rate <= 0.10, f"{tracker} clean jump_rate too high: {clean}"

        slow = by_key[("slow_noisy", tracker)]
        assert slow.link_yield >= 0.20, f"{tracker} slow_noisy yield collapsed: {slow}"
        assert slow.precision >= 0.40, f"{tracker} slow_noisy precision collapsed: {slow}"

        dense = by_key[("dense_ghost", tracker)]
        assert dense.link_yield >= 0.30, f"{tracker} dense_ghost yield collapsed: {dense}"
        assert dense.precision >= 0.40, f"{tracker} dense_ghost precision collapsed: {dense}"
