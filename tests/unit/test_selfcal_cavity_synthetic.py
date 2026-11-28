"""Self-calibration on a synthetic cavity (cameras on both sides, glass vectors of both signs) with the truth known."""

from __future__ import annotations

import copy
import shutil

import numpy as np
import pytest

from openptv2.calibration_consistency import self_calibrate_from_store
from openptv2.correspondences import match_correspondences_batch_parallel
from openptv2.loco import project_pixels
from openptv2.storage import RunStore
from openptv2.synthetic_rig import TruthRig, cavity_rig, synthetic_run, transformed

pytestmark = pytest.mark.unit
FRAMES = [1, 2, 3, 4]


def _drift(cals, scale, which=None, seed=3):
    rng = np.random.default_rng(seed)
    out = [copy.deepcopy(c) for c in cals]
    for i, c in enumerate(out):
        dp, da, dpp = rng.normal(0, 0.15 * scale, 3), rng.normal(0, 2e-4 * scale, 3), rng.normal(0, 0.003 * scale, 2)
        if which is None or i in which:
            c.set_pos(c.get_pos() + dp)
            c.set_angles(c.get_angles() + da)
            c.int_par.xh += dpp[0]
            c.int_par.yh += dpp[1]
    return out


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("cavity")
    rig = cavity_rig()
    synthetic_run(rig, d / "blobs.zarr", FRAMES, n_particles=900, seed=1)
    return rig, d


def _fit(world, model, tag, truth_rig=None, **kw):
    rig, d = world
    path = d / f"{tag}.zarr"
    if path.exists():
        shutil.rmtree(path)
    shutil.copytree(d / "blobs.zarr", path)
    match_correspondences_batch_parallel(frames=FRAMES, cpar=rig.cpar, cals=list(model), vpar=rig.vpar, zarr_store_path=str(path),
                                         n_workers=1, write_to_store=True)
    st = RunStore(str(path), mode="r")
    res = self_calibrate_from_store(st, FRAMES, list(model), rig.cpar, rig.vpar, min_yield=0, max_nfev=120, **kw)
    truth = truth_rig or rig
    pts = np.random.default_rng(11).uniform(-30, 30, (800, 3))
    pts = pts if truth_rig is None else pts  # points in the truth rig's own frame are produced by the caller when moved
    err = np.sqrt(np.nanmean([(project_pixels(pts, res.cals[c], rig.cpar) - truth.project_cam(c, pts)) ** 2 for c in range(4)]))
    return res, float(err)


def test_the_automatic_reference_camera_is_not_the_one_that_drifted(world):
    """Holding a camera fixed keeps its own errors for good and moves the others to agree with it. If camera 1 has drifted and is held
    (the old default), the fit makes the calibration worse in image space (1.03 -> 1.60 px in the study); holding the most consistent
    camera does not."""
    rig, _ = world
    model = _drift(rig.cals, 0.5, which=[0])
    auto, err_auto = _fit(world, model, "auto")
    held, err_held = _fit(world, model, "hold0", hold_cam=0)
    assert auto.hold_cam != 0
    assert max(auto.after.jump_um) < 10 and max(held.after.jump_um) < 10          # both are consistent ...
    assert err_auto < 0.7 * err_held, (err_auto, err_held)                         # ... only one is right


def test_a_half_turn_of_the_whole_rig_changes_nothing(world):
    """The rig rotated by 180 deg about y puts the +z cameras at -z (and their glass vectors with them). Identical blobs, same camera
    held: the self-calibration must give the same consistency and the same error."""
    rig, d = world
    rot = np.diag([-1.0, 1.0, -1.0])
    model = _drift(rig.cals, 0.35)
    base, err0 = _fit(world, model, "inv0", hold_cam=0)
    moved_model = transformed(TruthRig.from_cals(model, rig.cpar), rot, np.zeros(3)).cals
    truth_moved = transformed(rig, rot, np.zeros(3))
    moved, err1 = _fit(world, moved_model, "inv1", truth_rig=truth_moved, hold_cam=0)
    assert abs(max(base.after.jump_um) - max(moved.after.jump_um)) < 3.0
    assert abs(err0 - err1) < 0.1 * max(err0, 0.3), (err0, err1)


def test_glass_vectors_keep_their_sign_and_direction_on_both_sides(world):
    """Free glass distance must change only the length of the glass vector, for cameras with negative and with positive z."""
    rig, _ = world
    model = []
    for c in rig.cals:
        c = copy.deepcopy(c)
        g = np.asarray(c.get_glass_vec(), float)
        c.set_glass_vec(g / np.linalg.norm(g) * (np.linalg.norm(g) + 1.0))
        c.invalidate_mmlut()
        model.append(c)
    before = [np.asarray(c.get_glass_vec(), float) for c in model]
    res, _ = _fit(world, model, "glass", free_glass=True)
    for b, c in zip(before, res.cals):
        a = np.asarray(c.get_glass_vec(), float)
        assert np.sign(a[2]) == np.sign(b[2]) and np.allclose(a[:2], 0, atol=1e-9)
        assert np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)) > 1 - 1e-12
