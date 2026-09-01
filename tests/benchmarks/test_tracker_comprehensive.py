"""Decisive tracker benchmark: accuracy vs speed vs robustness.

Reuses the existing, tested benchmarking harness
(``openptv2.benchmarking``: ``ScenarioSpec``/``generate_scenario`` for
ground-truth trajectories with gaps/ghosts/turbulence, ``write_experiment``
to produce a runnable experiment folder, ``run_tracker`` to drive a real
tracker plugin, ``compute_identity_metrics`` for P/R/F1-style scoring)
instead of hand-rolling a second synthetic pipeline — see
tests/unit/test_benchmarking.py for the harness's own tests.

Decides the accuracy-vs-speed-vs-robustness trade-off across trackers at
controlled SNR, rather than the old fixed 5-frame smoke which could only say
"it links".

See docs/plans/2026-09-02-refactor-burgers-synthetic-tests.md Phase 5 (§10).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

import openptv2.benchmarking as bm

pytestmark = [pytest.mark.slow, pytest.mark.ci]

TRACKERS_UNDER_TEST = ["fast_3d", "priority_segment_3d"]
NOISE_LEVELS_MM = [0.0, 0.3, 0.6]


def _track_overrides(velocity: float) -> dict[str, float]:
    """Loose-but-bounded gates scaled to the scenario's velocity, mirroring
    tests/unit/test_benchmarking.py::test_runner_reconstructs_tracks."""
    dv = max(3.0, velocity * 3.0)
    return dict(
        dvxmin=-dv, dvxmax=dv, dvymin=-dv, dvymax=dv, dvzmin=-dv, dvzmax=dv,
        dacc=max(3.0, velocity * 3.0),
    )


def _run_and_score(
    tmp_path: Path, tracker: str, spec: bm.ScenarioSpec, first_frame: int = 10001
):
    """Generate a scenario, run one tracker, return (metrics, ms_per_frame)."""
    true_tracks, frame_gt = bm.generate_scenario(spec)
    rig = bm.make_standard_rig(refract=False)
    yaml_path = bm.write_experiment(rig, frame_gt, tmp_path, first_frame=first_frame)

    t0 = time.perf_counter()
    pred = bm.run_tracker(
        yaml_path, tracker, track_overrides=_track_overrides(spec.velocity)
    )
    dt_ms_per_frame = (time.perf_counter() - t0) * 1000.0 / spec.num_frames

    # generate_scenario numbers frames 0-based; run_tracker/write_experiment
    # number them from first_frame — shift before matching or every frame
    # miss and completeness reads a flat 0.0 regardless of tracking quality.
    true_tracks_abs = {
        pid: [(f + first_frame, x, y, z) for f, x, y, z in pts]
        for pid, pts in true_tracks.items()
    }
    # Matching tolerance must cover the injected detection noise itself, or
    # every point misses its true particle regardless of tracking quality.
    eps = max(0.5, 2.0 * spec.noise_mm)
    metrics = bm.compute_identity_metrics(true_tracks_abs, pred, eps=eps)
    return metrics, dt_ms_per_frame


@pytest.mark.parametrize("noise_mm", NOISE_LEVELS_MM)
@pytest.mark.parametrize("tracker", TRACKERS_UNDER_TEST)
def test_tracker_accuracy_vs_noise(tmp_path: Path, tracker: str, noise_mm: float):
    """Accuracy degrades gracefully with detection noise; a working tracker
    stays well above chance even at the noisiest level tested."""
    spec = bm.ScenarioSpec(
        num_particles=15,
        num_frames=10,
        velocity=1.0,
        velocity_jitter=0.2,
        noise_mm=noise_mm,
        seed=0,
    )
    metrics, ms_per_frame = _run_and_score(tmp_path, tracker, spec)

    assert ms_per_frame < 2000, f"{tracker}: {ms_per_frame:.1f} ms/frame (hang?)"
    # Completeness (fraction of each true track's points recovered) must not
    # collapse even at 0.6mm noise on a ~1mm/frame motion scenario.
    assert metrics.completeness > 0.5, (
        f"{tracker} @ noise={noise_mm}: completeness={metrics.completeness:.2f}"
    )


def test_tracker_speed_scaling(tmp_path: Path):
    """Speed scales sub-quadratically with particle density."""
    times_ms = {}
    for n in [10, 20, 40]:
        spec = bm.ScenarioSpec(
            num_particles=n, num_frames=8, velocity=1.0, seed=1
        )
        _, ms_per_frame = _run_and_score(tmp_path / f"n{n}", "fast_3d", spec)
        times_ms[n] = ms_per_frame
    # 4x density should cost less than 8x time (sub-quadratic slope).
    assert times_ms[40] < times_ms[10] * 8 + 1.0, times_ms


def test_tracker_robustness_gap_relinking(tmp_path: Path):
    """A tracker must keep recovering most of a track's points even with
    per-frame detection dropouts (regression surface for the double-claim
    bug in docs/plans/2026-08-27-backward-postprocess-double-claim-bug-plan.md)."""
    spec = bm.ScenarioSpec(
        num_particles=20,
        num_frames=12,
        velocity=1.0,
        velocity_jitter=0.2,
        gap_probability=0.1,
        seed=2,
    )
    metrics, _ = _run_and_score(tmp_path, "priority_segment_3d", spec)
    assert metrics.completeness > 0.4, metrics
    # No predicted track should be so fragmented it's meaningless.
    assert metrics.fragmentation < 5.0, metrics
