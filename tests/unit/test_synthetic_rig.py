"""The independent ray tracer (openptv2.synthetic_rig) against the library's radial-shift multimedia model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openptv2.loco import project_pixels
from openptv2.recorrespond import load_parameters
from openptv2.synthetic_rig import TruthRig

pytestmark = pytest.mark.unit

YAML = Path(__file__).resolve().parents[2] / "test_data" / "test_cavity_small" / "parameters_Run1.yaml"


@pytest.fixture(scope="module")
def rig_pts():
    cpar, _vpar, cals, _ = load_parameters(YAML)
    pts = np.random.default_rng(0).uniform(-4, 4, (200, 3))
    return TruthRig.from_cals(cals, cpar), cals, cpar, pts


def test_ideal_layers_agree_with_the_library_model(rig_pts):
    """Snell's law traced in 3D equals the library's flat-layer model (air | glass | water) to a thousandth of a pixel."""
    rig, cals, cpar, pts = rig_pts
    px = rig.project(pts)
    for c in range(len(cals)):
        ref = project_pixels(pts, cals[c], cpar)
        ok = np.isfinite(px[:, c, 0])
        assert ok.sum() > 100
        assert np.abs(px[ok, c] - ref[ok]).max() < 2e-3


def test_a_different_truth_moves_the_image_by_a_known_amount(rig_pts):
    rig, cals, cpar, pts = rig_pts
    base = rig.project(pts)
    for name, other in (("fluid index", rig.with_(n_water=rig.n_water + 0.02)),
                        ("front tilt", rig.with_(tilt_front=[(0.005, 0.0)] * 4)),
                        ("back tilt", rig.with_(tilt_back=[(0.0, 0.005)] * 4))):
        d = np.abs(other.project(pts) - base)
        assert np.nanmax(d) > 0.01, name          # a real effect ...
        assert np.nanmax(d) < 5.0, name           # ... but a small perturbation, not garbage


def test_model_cpar_carries_the_rig_layers(rig_pts):
    rig, _cals, _cpar, _pts = rig_pts
    cp = rig.with_(n_water=1.4, glass_thickness=9.0).model_cpar() if False else rig.with_(n_water=1.4, glass_thickness=9.0)
    mm = cp.model_cpar().mm
    assert float(mm.n3) == 1.4 and float(np.atleast_1d(mm.d)[0]) == 9.0


def test_with_copies_and_does_not_touch_the_original(rig_pts):
    rig, _cals, _cpar, _pts = rig_pts
    other = rig.with_(glass_thickness=11.0)
    other.cals[0].int_par.xh += 1.0
    assert rig.glass_thickness != 11.0
    assert rig.cals[0].int_par.xh != other.cals[0].int_par.xh


def _rot(axis, angle):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def test_cavity_rig_has_cameras_on_both_sides_and_sees_the_volume():
    from openptv2.synthetic_rig import cavity_rig

    rig = cavity_rig()
    z = [float(c.get_pos()[2]) for c in rig.cals]
    gz = [float(c.get_glass_vec()[2]) for c in rig.cals]
    assert z[0] < 0 and z[1] < 0 and z[2] > 0 and z[3] > 0
    assert gz[0] < 0 and gz[1] < 0 and gz[2] > 0 and gz[3] > 0          # the glass vector points to the camera's side
    pts = np.random.default_rng(1).uniform(-30, 30, (400, 3))
    assert np.isfinite(rig.project(pts)).all(axis=(1, 2)).mean() > 0.95


@pytest.mark.parametrize("axis,angle,shift", [((0, 1, 0), np.pi, (0, 0, 0)), ((1, 0, 0), np.pi / 2, (0, 0, 0)),
                                              ((1, 2, 3), 0.7, (10.0, -20.0, 7.0)), ((0, 0, 1), 0.0, (5.0, 0.0, -30.0))])
def test_moving_the_whole_rig_changes_no_image(axis, angle, shift):
    """A rotation by 180 deg about y puts the +z cameras at -z (and their glass vectors with them); images must not change."""
    from openptv2.synthetic_rig import cavity_rig, transformed

    rig = cavity_rig()
    r, t = _rot(axis, angle), np.asarray(shift, float)
    moved = transformed(rig, r, t)
    pts = np.random.default_rng(2).uniform(-25, 25, (300, 3))
    a, b = rig.project(pts), moved.project(pts @ r.T + t)
    ok = np.isfinite(a).all(axis=(1, 2)) & np.isfinite(b).all(axis=(1, 2))
    assert ok.sum() > 200
    assert np.abs(a[ok] - b[ok]).max() < 2e-3


def test_synthetic_run_scores_the_truth_matching(tmp_path):
    from openptv2.correspondences import match_correspondences_batch_parallel
    from openptv2.storage import RunStore
    from openptv2.synthetic_rig import cavity_rig, score_matching, synthetic_run

    rig = cavity_rig()
    truth = synthetic_run(rig, tmp_path / "r.zarr", [1], n_particles=300, seed=3)
    match_correspondences_batch_parallel(frames=[1], cpar=rig.cpar, cals=rig.cals, vpar=rig.vpar, zarr_store_path=str(tmp_path / "r.zarr"),
                                         n_workers=1, write_to_store=True)
    sc = score_matching(RunStore(str(tmp_path / "r.zarr"), mode="r"), [1], truth)
    assert sc["possible"] > 50 and sc["recall"] > 0.8 and sc["err_med"] < 0.05, sc
