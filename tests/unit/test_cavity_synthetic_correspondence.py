"""Phase 3: correspondence quad-uniqueness on the on-demand cavity factory.

8 frames, n=120, accel_sigma 0.4, pixel_noise 0.25 — probes the SNR lever
migrated from tests/batch/test_burgers_synthetic.py::test_burgers_3d_trajectory_vs_res_orig
(3D trajectory round-trip vs committed res_orig ground truth).

See docs/plans/2026-09-02-refactor-burgers-synthetic-tests.md Phase 3.
"""

from pathlib import Path

import numpy as np
import pytest

from openptv2.storage import RunStore
from tests.helpers.synthetic_scene import make_cavity_scene

pytestmark = [pytest.mark.ci]


def test_cavity_synthetic_correspondence_quad_uniqueness(tmp_path: Path):
    """Each 3D correspondence claims a given (cam, target) pair at most once
    per frame — the quad-uniqueness invariant `correspondences()` must hold,
    and the regression `forward_backward` double-claim (2026-08-27 plan) must
    not reappear once a scene has real accel + pixel noise."""
    n_frames, n_particles = 8, 120
    scene = make_cavity_scene(
        tmp_path,
        n_frames=n_frames,
        n_particles=n_particles,
        accel_sigma=0.4,
        pixel_noise=0.25,
        seed=6,
    )
    store = RunStore(scene / "res" / "run.zarr", mode="r")
    assert len(store.frames()) == n_frames

    for f in range(n_frames):
        pos, cam_ids = store.read_correspondences(10001 + f)
        assert pos.shape[1] == 3
        assert np.all(np.isfinite(pos))
        # quad-uniqueness: no (cam, pnr) pair claimed by two different 3D points
        for cam in range(cam_ids.shape[1]):
            claimed = cam_ids[:, cam]
            visible = claimed[claimed >= 0]
            assert len(set(visible.tolist())) == len(visible), (
                f"frame {f} cam {cam}: duplicate target claim"
            )


def test_cavity_synthetic_correspondence_recovers_all_particles(tmp_path: Path):
    """With modest accel + pixel noise, correspondence count should track
    n_particles closely (recall-based, mirrors the migrated burgers test's
    'must recover every ground-truth position' assertion)."""
    n_frames, n_particles = 8, 120
    scene = make_cavity_scene(
        tmp_path,
        n_frames=n_frames,
        n_particles=n_particles,
        accel_sigma=0.4,
        pixel_noise=0.25,
        seed=7,
    )
    store = RunStore(scene / "res" / "run.zarr", mode="r")
    for f in range(n_frames):
        pos, _ = store.read_correspondences(10001 + f)
        assert len(pos) == n_particles
