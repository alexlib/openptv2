"""Ablation benchmark: isolate peeling / ILP / temporal-smoothing contributions.

Runs the unified tracker on a moderate synthetic scenario with each lever
toggled, reporting purity/completeness/median length so the marginal value of
each is visible.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

import openptv2.benchmarking as bm


def _run(tmp_path: Path, cfg_overrides: dict):
    spec = bm.ScenarioSpec(
        num_particles=60,
        num_frames=12,
        velocity=1.2,
        velocity_jitter=0.25,
        noise_mm=0.1,
        ghost_ratio=0.3,
        seed=9,
        flow_type="turbulent",
    )
    true_tracks, frame_gt = bm.generate_scenario(spec)
    rig = bm.make_standard_rig(refract=False)
    yaml_path = bm.write_experiment(rig, frame_gt, tmp_path, first_frame=10001)
    # Inject track_cfg overrides by writing them into the yaml's track section.
    import yaml
    data = yaml.safe_load(yaml_path.read_text())
    data.setdefault("track", {}).update(cfg_overrides)
    yaml_path.write_text(yaml.dump(data))
    t0 = time.perf_counter()
    pred = bm.run_tracker(yaml_path, "unified")
    elapsed = time.perf_counter() - t0
    true_abs = {pid: [(f + 10001, x, y, z) for f, x, y, z in pts] for pid, pts in true_tracks.items()}
    idm = bm.compute_identity_metrics(true_abs, pred, eps=max(0.5, 2 * spec.noise_mm))
    lengths = [len(p) for p in pred.values()]
    return {
        "n": len(pred),
        "med_len": float(np.median(lengths)) if lengths else 0.0,
        "completeness": idm.completeness,
        "purity": idm.purity,
        "time": elapsed,
    }


@pytest.mark.slow
def test_ablation_peeling(tmp_path: Path):
    a = _run(tmp_path / "peel_on", {"peeling": True})
    b = _run(tmp_path / "peel_off", {"peeling": False})
    print(f"peeling ON : {a}")
    print(f"peeling OFF: {b}")
    # peeling should not crash; purity should be at least comparable
    assert a["purity"] >= 0.5 and b["purity"] >= 0.5


@pytest.mark.slow
def test_ablation_temporal_smoothing(tmp_path: Path):
    a = _run(tmp_path / "ts_on", {"temporal_smoothing": True})
    b = _run(tmp_path / "ts_off", {"temporal_smoothing": False})
    print(f"smoothing ON : {a}")
    print(f"smoothing OFF: {b}")
    assert a["purity"] >= 0.5 and b["purity"] >= 0.5