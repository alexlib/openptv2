"""Phase 3: track3d on the on-demand cavity factory, exported to legacy
rt_is/*_targets and run through the real Tracker class.

12 frames, turb_sigma 0.6, pixel_noise 0.5 — migrated from
tests/unit/test_track.py::test_burgers (fixed 5-frame `test_data/burgers`).
Uses test_cavity's real track params (dacc, dangle, dv*) as the gate, so the
SNR lever (spacing 3.8mm vs motion 0.3mm) is actually exercised.

See docs/plans/2026-09-02-refactor-burgers-synthetic-tests.md Phase 3.
"""

import os
from pathlib import Path

import pytest

from openptv2.algorithms.parameters import ControlPar, SequencePar, TrackPar, VolumePar
from openptv2.storage import RunStore
from openptv2.storage.legacy import export_run
from tests.helpers.synthetic_scene import load_cavity_calibration, make_cavity_scene

pytestmark = [pytest.mark.ci]

CAVITY_YAML = str(
    Path(__file__).resolve().parents[2] / "test_data/test_cavity/parameters.yaml"
)


def test_track_cavity_synthetic_forward_tracking(tmp_path: Path):
    """Forward tracking on a 12-frame turbulent+noisy synthetic scene must
    link most particles frame-to-frame (npart tracks n_particles*n_frames
    order of magnitude, nlinks stays a large fraction of npart)."""
    from openptv2.tracker import Tracker

    n_frames, n_particles = 12, 80
    scene = make_cavity_scene(
        tmp_path,
        n_frames=n_frames,
        n_particles=n_particles,
        spacing_mm=3.8,
        motion_mm=0.3,
        turb_sigma=0.6,
        pixel_noise=0.5,
        seed=8,
    )
    store = RunStore(scene / "res" / "run.zarr", mode="r")
    export_run(store, scene)

    cpar = ControlPar.from_yaml(CAVITY_YAML)
    vpar = VolumePar.from_yaml(CAVITY_YAML)
    tpar = TrackPar.from_yaml(CAVITY_YAML)
    _, cals = load_cavity_calibration("test_data/test_cavity")
    spar = SequencePar(
        num_cams=cpar.num_cams,
        img_base_name=["img/cam1.", "img/cam2.", "img/cam3.", "img/cam4."],
        first=10001,
        last=10001 + n_frames - 1,
    )

    original = os.getcwd()
    try:
        os.chdir(scene)
        tracker = Tracker(cpar, vpar, tpar, spar, cals)
        tracker.full_forward_3d()

        assert tracker.npart > 0
        assert tracker.nlinks > 0
        # Most particles should link at least once given smooth turbulence
        # (not a tight bound — this is an SNR smoke gate, not a golden number).
        assert tracker.nlinks >= 0.5 * tracker.npart, (
            f"npart={tracker.npart} nlinks={tracker.nlinks}"
        )
    finally:
        os.chdir(original)
