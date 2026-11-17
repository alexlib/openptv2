"""Realistic-density benchmarks for the unified tracker.

Uses synthetic scenarios at moderate densities (50-200 particles) that mirror
real PTV datasets (burgers, cavity) rather than the adversarial 1000-particle
stress case. Reports track counts, median length, jump rate, purity.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

import openptv2.benchmarking as bm
from openptv2.benchmarking.metrics import compute_identity_metrics


@pytest.mark.slow
def test_unified_moderate_density(tmp_path: Path):
    spec = bm.ScenarioSpec(
        num_particles=120,
        num_frames=20,
        velocity=1.2,
        velocity_jitter=0.2,
        noise_mm=0.05,
        ghost_ratio=0.15,
        seed=21,
        flow_type="turbulent",
    )
    true_tracks, frame_gt = bm.generate_scenario(spec)
    rig = bm.make_standard_rig(refract=False)
    yaml_path = bm.write_experiment(rig, frame_gt, tmp_path, first_frame=10001)
    t0 = time.perf_counter()
    pred = bm.run_tracker(yaml_path, "unified")
    elapsed = time.perf_counter() - t0
    lengths = [len(p) for p in pred.values()]
    true_abs = {pid: [(f + 10001, x, y, z) for f, x, y, z in pts] for pid, pts in true_tracks.items()}
    idm = bm.compute_identity_metrics(true_abs, pred, eps=max(0.5, 2 * spec.noise_mm))
    print(f"moderate: n_tracks={len(pred)} med_len={np.median(lengths):.1f} "
          f"completeness={idm.completeness:.3f} purity={idm.purity:.3f} time={elapsed:.1f}s")
    assert len(pred) > 0
    assert np.median(lengths) >= 3.0
    assert idm.completeness >= 0.8
    assert idm.purity >= 0.85


@pytest.mark.slow
def test_unified_low_density(tmp_path: Path):
    spec = bm.ScenarioSpec(
        num_particles=40,
        num_frames=15,
        velocity=1.0,
        velocity_jitter=0.15,
        noise_mm=0.0,
        ghost_ratio=0.0,
        seed=22,
    )
    true_tracks, frame_gt = bm.generate_scenario(spec)
    rig = bm.make_standard_rig(refract=False)
    yaml_path = bm.write_experiment(rig, frame_gt, tmp_path, first_frame=10001)
    pred = bm.run_tracker(yaml_path, "unified")
    lengths = [len(p) for p in pred.values()]
    assert len(pred) > 0
    assert np.median(lengths) >= 5.0