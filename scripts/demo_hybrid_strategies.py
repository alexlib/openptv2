#!/usr/bin/env python3
"""Demo & Benchmark comparison of Single-Pass vs Hybrid Cascading Strategies.

Compares:
1. Baseline Default Single-Pass: priority_segment_3d
2. Strategy 1: Forward-Fast / Backward-GMM Hybrid
   (Fast priority_segment_3d forward pass + predictive_gmm_3d backward pass on unlinked residue)
3. Strategy 2: Two-Scale Velocity Cascading
   (Coarse nearest_hungarian_3d pass + Fine predictive_gmm_3d pass on residual particles)

Emits unified performance metrics (Precision, Recall/Yield, Purity, PMT%, Speed).
"""

from __future__ import annotations

import shutil
import tempfile
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark_utils as bu
import openptv2.benchmarking as bm


def _isolate_and_set_n_frames(src: Path, first: int, n_frames: int) -> tuple[Path, Path]:
    import yaml
    run_dir, yaml_run = bu._isolate_run_dir(src)
    data = yaml.safe_load(yaml_run.read_text())
    data["sequence"]["first"] = first
    data["sequence"]["last"] = first + n_frames - 1
    yaml_run.write_text(yaml.dump(data))
    return run_dir, yaml_run


def run_strategy_single_pass(
    src: Path = bu.SRC, first: int = bu.FIRST, n_frames: int = bu.N_FRAMES
) -> tuple[dict, float]:
    """Strategy 0: Baseline Single Pass using priority_segment_3d."""
    run_dir, yaml_run = _isolate_and_set_n_frames(src, first, n_frames)
    try:
        t0 = time.perf_counter()
        pred = bm.run_tracker(yaml_run, "priority_segment_3d", track_overrides=bu.BASE_OVERRIDES)
        dt = time.perf_counter() - t0
        pred0 = {k: [(f - first, x, y, z) for (f, x, y, z) in v] for k, v in pred.items()}
        return pred0, dt
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def run_strategy_1_forward_fast_backward_gmm(
    src: Path = bu.SRC, first: int = bu.FIRST, n_frames: int = bu.N_FRAMES
) -> tuple[dict, float]:
    """Strategy 1: Forward-Fast / Backward-GMM Hybrid."""
    run_dir, yaml_run = _isolate_and_set_n_frames(src, first, n_frames)
    try:
        t0 = time.perf_counter()
        # 1. Forward fast pass
        pred_fwd = bm.run_tracker(yaml_run, "priority_segment_3d", track_overrides=bu.BASE_OVERRIDES)

        # 2. Backward GMM pass for gap filling and tight turn recovery
        gmm_overrides = dict(bu.BASE_OVERRIDES)
        gmm_overrides["reverse"] = True
        pred_bwd = bm.run_tracker(yaml_run, "predictive_gmm_3d", track_overrides=gmm_overrides)

        # 3. Merge tracks: start with forward links, add non-overlapping backward links
        merged_tracks = {}
        track_id_counter = 0

        for t_id, pts in pred_fwd.items():
            if len(pts) >= 2:
                merged_tracks[track_id_counter] = [(f - first, x, y, z) for (f, x, y, z) in pts]
                track_id_counter += 1

        # Track existing frame-point signatures to avoid duplicate links
        existing_keys = set()
        for pts in merged_tracks.values():
            for f, x, y, z in pts:
                existing_keys.add((f, round(x, 2), round(y, 2), round(z, 2)))

        for t_id, pts in pred_bwd.items():
            if len(pts) >= 2:
                unique_pts = [(f - first, x, y, z) for (f, x, y, z) in pts
                              if (f - first, round(x, 2), round(y, 2), round(z, 2)) not in existing_keys]
                if len(unique_pts) >= 2:
                    merged_tracks[track_id_counter] = sorted(unique_pts, key=lambda p: p[0])
                    track_id_counter += 1

        dt = time.perf_counter() - t0
        return merged_tracks, dt
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def run_strategy_2_two_scale_velocity_cascading(
    src: Path = bu.SRC, first: int = bu.FIRST, n_frames: int = bu.N_FRAMES
) -> tuple[dict, float]:
    """Strategy 2: Two-Scale Velocity Cascading."""
    run_dir, yaml_run = _isolate_and_set_n_frames(src, first, n_frames)
    try:
        t0 = time.perf_counter()
        # 1. Coarse high-velocity pass
        coarse_overrides = dict(bu.BASE_OVERRIDES)
        coarse_overrides["dvxmax"] = 12.0
        coarse_overrides["dvxmin"] = -12.0
        coarse_overrides["dvymax"] = 12.0
        coarse_overrides["dvymin"] = -12.0
        coarse_overrides["dvzmax"] = 12.0
        coarse_overrides["dvzmin"] = -12.0
        pred_coarse = bm.run_tracker(yaml_run, "nearest_hungarian_3d", track_overrides=coarse_overrides)

        # 2. Fine pass for micro-eddies and tight turns
        fine_overrides = dict(bu.BASE_OVERRIDES)
        fine_overrides["dvxmax"] = 4.0
        fine_overrides["dvxmin"] = -4.0
        fine_overrides["dvymax"] = 4.0
        fine_overrides["dvymin"] = -4.0
        fine_overrides["dvzmax"] = 4.0
        fine_overrides["dvzmin"] = -4.0
        pred_fine = bm.run_tracker(yaml_run, "predictive_gmm_3d", track_overrides=fine_overrides)

        # 3. Combine non-conflicting trajectories
        merged_tracks = {}
        track_id_counter = 0

        for t_id, pts in pred_coarse.items():
            if len(pts) >= 2:
                merged_tracks[track_id_counter] = [(f - first, x, y, z) for (f, x, y, z) in pts]
                track_id_counter += 1

        existing_keys = set()
        for pts in merged_tracks.values():
            for f, x, y, z in pts:
                existing_keys.add((f, round(x, 2), round(y, 2), round(z, 2)))

        for t_id, pts in pred_fine.items():
            if len(pts) >= 2:
                unique_pts = [(f - first, x, y, z) for (f, x, y, z) in pts
                              if (f - first, round(x, 2), round(y, 2), round(z, 2)) not in existing_keys]
                if len(unique_pts) >= 2:
                    merged_tracks[track_id_counter] = sorted(unique_pts, key=lambda p: p[0])
                    track_id_counter += 1

        dt = time.perf_counter() - t0
        return merged_tracks, dt
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def compare_all_strategies(src: Path = bu.SRC, first: int = bu.FIRST, n_frames: int = bu.N_FRAMES) -> list[dict]:
    """Run all strategies and compute combined benchmark metrics."""
    frames = bu.read_gt_frames(src, first, n_frames)
    tt = bu.build_true_tracks(frames, first)
    ghosts = bu.build_ghost_frames(frames, first)

    strategies = [
        ("Baseline (Single Pass)", run_strategy_single_pass),
        ("Strategy 1 (Fwd-Fast/Bwd-GMM)", run_strategy_1_forward_fast_backward_gmm),
        ("Strategy 2 (Two-Scale Cascading)", run_strategy_2_two_scale_velocity_cascading),
    ]

    rows = []
    n_steps = max(1, n_frames - 1)

    for name, fn in strategies:
        pred0, dt = fn(src, first, n_frames)
        metrics = bu.combined_metrics(tt, pred0, eps=1.0, ghosts=ghosts)
        ms_per_frame = 1000.0 * dt / n_steps
        rows.append({
            "strategy": name,
            "ms_per_frame": ms_per_frame,
            **metrics,
        })
    return rows


def print_comparison_table(rows: list[dict]) -> None:
    header = (f"{'strategy':<32} | {'precision':>9} | {'recall':>7} | "
              f"{'ghost%':>7} | {'F':>5} | {'C':>5} | {'purity':>6} | {'pmt%':>6} | "
              f"{'ms/frame':>9}")
    print("\n=== HYBRID TRACKING STRATEGIES COMPARISON BENCHMARK ===")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['strategy']:<32} | {r['precision']:>9.3f} | "
            f"{r['yield_recall']:>7.3f} | {100 * r['ghost_capture_rate']:>6.2f}% | "
            f"{r['fragmentation']:>5.2f} | {r['completeness']:>5.2f} | "
            f"{r['purity']:>6.3f} | {r['pmt']:>5.1f}% | {r['ms_per_frame']:>9.1f}"
        )


def main() -> None:
    rows = compare_all_strategies()
    print_comparison_table(rows)


if __name__ == "__main__":
    main()
