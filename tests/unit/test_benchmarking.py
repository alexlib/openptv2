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


def test_scenario_turbulent_flow():
    """Turbulent flow produces curved (non-straight) trajectories."""
    spec = bm.ScenarioSpec(
        num_particles=20, num_frames=40, velocity=2.0, velocity_jitter=1.0,
        flow_type="turbulent", seed=4,
    )
    tt, fg = bm.generate_scenario(spec)
    # tracks should be long and their per-frame step directions vary (curved)
    assert all(len(v) == 40 for v in tt.values())
    # measure angular variance of successive steps for one track
    track = tt[next(iter(tt))]
    pts = np.array([p[1:] for p in track])
    steps = np.diff(pts, axis=0)
    steps = steps / np.linalg.norm(steps, axis=1, keepdims=True)
    # dot of consecutive unit steps should not all be ~1 (not a straight line)
    dots = np.clip(np.sum(steps[:-1] * steps[1:], axis=1), -1, 1)
    assert np.mean(dots) < 0.95


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


def test_metrics_one_to_one_no_double_claim():
    """Two predicted points near one true particle must not both match it.

    Regression for the old unconstrained-nearest-neighbour _match_frame,
    which let two predicted tracks both claim the same true particle in a
    frame (many-to-one). With eps wide enough to cover a second, more
    distant true particle too, a many-to-one match would send the extra
    predicted point to the wrong true particle instead of leaving it
    unmatched -- inflating purity/pmt on data that is actually ambiguous.
    """
    tt = {
        0: [(0, 0.0, 0.0, 0.0)],
        1: [(0, 1.0, 0.0, 0.0)],
    }
    # Both predicted points sit closer to true particle 0 than to 1.
    pred = {
        10: [(0, 0.05, 0.0, 0.0)],
        11: [(0, 0.15, 0.0, 0.0)],
    }
    m = bm.compute_identity_metrics(tt, pred, eps=0.6)
    # Only one of the two predicted points can be assigned to true id 0;
    # the other must go unmatched (true id 1 is too far for either).
    assert m.n_correct_tracks == 1


def test_metrics_ghost_capture_rate():
    """A predicted point that only matches a ghost is reported as a capture."""
    tt = {0: [(f, float(f), 0.0, 0.0) for f in range(5)]}
    pred = {
        0: [(f, float(f), 0.0, 0.0) for f in range(5)],  # correctly tracks id 0
        1: [(f, float(f), 5.0, 0.0) for f in range(5)],  # only near the ghost
    }
    ghosts = {f: np.array([[float(f), 5.05, 0.0]]) for f in range(5)}
    m = bm.compute_identity_metrics(tt, pred, eps=0.5, ghost_pos_by_frame=ghosts)
    assert m.n_ghost_captures == 5
    assert m.ghost_capture_rate == pytest.approx(5 / 10)


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


def test_track_lifetime_distribution():
    tracks = {
        0: [(f, float(f), 0.0, 0.0) for f in range(40)],  # long
        1: [(f, 0.0, float(f), 0.0) for f in range(15)],  # medium
        2: [(0, 0.0, 0.0, 0.0), (1, 1.0, 0.0, 0.0)],  # short
    }
    d = bm.track_lifetime_distribution(tracks)
    assert d["n_tracks"] == 3
    assert d["mean_track_length"] == pytest.approx((40 + 15 + 2) / 3)
    assert d["frac_tracks_over_10"] == pytest.approx(2 / 3)
    assert d["frac_tracks_over_30"] == pytest.approx(1 / 3)


def test_track_lifetime_distribution_empty():
    d = bm.track_lifetime_distribution({})
    assert d["n_tracks"] == 0
    assert d["mean_track_length"] == 0.0


def test_acceleration_kurtosis_zero_for_straight_line():
    """Constant velocity -> zero acceleration everywhere -> undefined (NaN)
    kurtosis, not a division-by-zero crash."""
    tracks = {0: [(f, float(f), 0.0, 0.0) for f in range(20)]}
    k, n = bm.acceleration_kurtosis(tracks)
    assert n > 0
    assert np.isnan(k)


def test_acceleration_kurtosis_detects_a_single_outlier():
    """A false-link-style single acceleration spike in an otherwise smooth
    track must inflate K_a well above the Gaussian value of 3 -- this is
    the whole point of the metric (docs/lagrangian_turbulence_quality_guide.md
    section B): precision can look fine while K_a exposes the outlier."""
    pts = [(f, float(f) + (5.0 if f == 20 else 0.0), 0.0, 0.0) for f in range(40)]
    k, n = bm.acceleration_kurtosis({0: pts})
    assert n > 0
    assert k > 10.0


def test_acceleration_kurtosis_skips_gap_bridged_triples():
    """A finite-difference triple spanning a bridged gap (frames 0,1,3 --
    not consecutive) must not be treated as a real acceleration sample."""
    tracks = {0: [(0, 0.0, 0.0, 0.0), (1, 1.0, 0.0, 0.0), (3, 10.0, 0.0, 0.0)]}
    k, n = bm.acceleration_kurtosis(tracks)
    assert n == 0
    assert np.isnan(k)


def test_compute_physics_metrics_bundles_both():
    tracks = {0: [(f, float(f), 0.0, 0.0) for f in range(15)]}
    m = bm.compute_physics_metrics(tracks)
    assert isinstance(m, bm.PhysicsMetrics)
    assert m.n_tracks == 1
    assert m.mean_track_length == 15.0
    d = m.to_dict()
    assert set(d) == {
        "mean_track_length", "frac_tracks_over_10", "frac_tracks_over_30",
        "n_tracks", "acceleration_kurtosis", "n_acceleration_samples",
    }
