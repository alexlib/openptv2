"""Unit tests for benchmark.datawriter, camera_rig, experiment, runner, metrics."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

import openptv2.benchmarking as bm
from openptv2.algorithms.tracking_frame_buf import Frame


def test_rig_projects_to_sensor():
    """A spread of points must land on-sensor for all cameras."""
    rig = bm.make_standard_rig(refract=False)
    rng = np.random.default_rng(0)
    pts = rng.uniform(-40, 40, size=(200, 3))
    for px in bm.project_to_pixels(rig, pts):
        frac = ((px[:, 0] > 0) & (px[:, 0] < 1280) & (px[:, 1] > 0) & (px[:, 1] < 1024)).mean()
        assert frac > 0.9


def test_rig_refraction_differs_from_air():
    """Refraction must change the projection vs all-air."""
    rng = np.random.default_rng(1)
    pts = rng.uniform(-30, 30, size=(50, 3))
    air = bm.make_standard_rig(refract=False)
    refr = bm.make_standard_rig(refract=True)
    p_air = bm.project_to_pixels(air, pts)
    p_ref = bm.project_to_pixels(refr, pts)
    for cam in range(4):
        assert np.abs(p_air[cam] - p_ref[cam]).mean() > 0.5


def test_scenario_basic():
    """Scenario produces the requested number of true tracks."""
    spec = bm.ScenarioSpec(num_particles=10, num_frames=20, velocity=1.0, seed=1)
    tt, fg = bm.generate_scenario(spec)
    assert len(tt) == 10
    assert len(fg) == 20


def test_scenario_entering_leaving():
    """Entering/leaving particles should produce tracks that enter/exit."""
    spec = bm.ScenarioSpec(
        num_particles=5, num_frames=30, velocity=1.0,
        entering_particles=2, leaving_particles=2, seed=2,
    )
    tt, fg = bm.generate_scenario(spec)
    # entering/leaving tracks have fewer than full-length points
    n_entered = sum(1 for v in tt.values() if len(v) < 30)
    assert n_entered >= 2


def test_scenario_gaps_and_ghosts():
    """Gaps drop particles, ghosts add pid -1."""
    spec = bm.ScenarioSpec(
        num_particles=10, num_frames=15, gap_probability=0.5, ghost_ratio=0.2, seed=3,
    )
    tt, fg = bm.generate_scenario(spec)
    # some frame should have fewer visible than true particles OR ghosts
    some_ghost = any(any(p[0] < 0 for p in fg[f]) for f in fg)
    assert some_ghost


def test_write_dataset_readable_by_frame():
    """Written rt_is/targets must be readable by the Frame class."""
    spec = bm.ScenarioSpec(num_particles=15, num_frames=10, velocity=1.0, seed=3)
    tt, fg = bm.generate_scenario(spec)
    rig = bm.make_standard_rig(refract=False)
    d = Path(tempfile.mkdtemp())
    yaml = bm.write_experiment(rig, fg, d, first_frame=10001)
    assert yaml.exists()

    res = str(d / "res" / "rt_is")
    f = Frame(num_cams=4, max_targets=30000)
    ok = f.read(
        res, "", prio_file_base=str(d / "res" / "added"),
        target_file_base=[str(d / "img" / f"cam{i + 1}") for i in range(4)],
        frame_num=10001,
    )
    assert ok
    assert f.num_parts == len(fg[0])


def test_runner_reconstructs_tracks():
    """fast_3d should reconstruct all full-length trajectories."""
    spec = bm.ScenarioSpec(
        num_particles=12, num_frames=10, velocity=1.0,
        gap_probability=0.0, noise_mm=0.0, seed=3,
    )
    tt, fg = bm.generate_scenario(spec)
    rig = bm.make_standard_rig(refract=False)
    d = Path(tempfile.mkdtemp())
    yaml = bm.write_experiment(rig, fg, d, first_frame=10001)

    pred = bm.run_tracker(
        yaml, "fast_3d",
        track_overrides=dict(dvxmax=3.0, dvxmin=-3.0, dvymax=3.0, dvymin=-3.0,
                             dvzmax=3.0, dvzmin=-3.0, dacc=3.0),
    )
    # every predicted track should be full length
    lenses = [len(v) for v in pred.values()]
    assert all(l == 10 for l in lenses), lenses


def test_metrics_perfect():
    """Perfect reconstruction gives F=1, C=1, purity=1, pmt=100."""
    def mk(tid, x0):
        return [(f, x0 + f, 0.0, 0.0) for f in range(10)]
    tt = {0: mk(0, 0.0), 1: mk(1, 5.0)}
    pred = {0: mk(0, 0.0), 1: mk(1, 5.0)}
    m = bm.compute_identity_metrics(tt, pred, eps=0.5)
    assert m.fragmentation == 1.0
    assert m.completeness == 1.0
    assert m.purity == 1.0
    assert m.pmt == 100.0


def test_metrics_fragmented():
    """A true track split in two raises fragmentation > 1."""
    def mk(x0):
        return [(f, x0 + f, 0.0, 0.0) for f in range(10)]
    tt = {0: mk(0.0)}
    pred = {0: mk(0.0)[:5], 10: mk(0.0)[5:]}
    m = bm.compute_identity_metrics(tt, pred, eps=0.5)
    assert m.fragmentation > 1.0
    assert m.completeness > 0.95


def test_metrics_wrong_link():
    """A track that jumps particles should have low purity/pmt."""
    tt = {0: [(f, f, 0.0, 0.0) for f in range(10)],
          1: [(f, 5.0 + f, 0.0, 0.0) for f in range(10)]}
    # pred track 0 = particle 0 for 5 frames, then jumps to particle 1
    pred = {0: [(f, 0.1 + f, 0.0, 0.0) for f in range(5)]
            + [(f, 5.1 + f, 0.0, 0.0) for f in range(5, 10)]}
    m = bm.compute_identity_metrics(tt, pred, eps=0.5)
    assert m.purity < 0.6
    assert m.pmt < 100.0
