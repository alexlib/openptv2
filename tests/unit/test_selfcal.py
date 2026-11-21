"""Joint self-calibration (openptv2.selfcal): a known miscalibration is found again from tracer observations."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from openptv2.loco import loco_residuals, project_pixels
from openptv2.recorrespond import load_parameters
from openptv2.selfcal import joint_self_calibrate

pytestmark = pytest.mark.unit

YAML = Path(__file__).resolve().parents[2] / "test_data" / "test_cavity_small" / "parameters_Run1.yaml"


@pytest.fixture(scope="module")
def rig():
    cpar, vpar, cals, _ = load_parameters(YAML)
    rng = np.random.default_rng(2)
    pts = rng.uniform(-10, 10, (4000, 3))
    px = np.stack([project_pixels(pts, c, cpar) for c in cals], axis=1)
    ok = np.all((px[:, :, 0] > 5) & (px[:, :, 0] < cpar.imx - 5) & (px[:, :, 1] > 5) & (px[:, :, 1] < cpar.imy - 5), axis=1)
    return cpar, vpar, cals, pts[ok][:400], px[ok][:400]


def _perturbed(cals):
    out = [copy.deepcopy(c) for c in cals]
    out[1].int_par.xh += 0.02
    out[2].added_par.p1 += 4e-4
    out[3].int_par.yh -= 0.015
    out[3].added_par.p2 += 3e-4
    return out


def test_a_known_miscalibration_is_corrected(rig):
    cpar, vpar, cals, _pts, px = rig
    bad = _perturbed(cals)
    before = np.nanmedian(loco_residuals(px, cpar, bad, vpar))
    from openptv2.loco import pixels_to_flat  # noqa: F401
    from openptv2.orientation import point_positions

    flat = np.stack([pixels_to_flat(px[:, c], bad[c], cpar) for c in range(4)], axis=1)
    x0 = np.asarray(point_positions(flat, cpar, bad, vpar, method="lsq")[0], float)
    res = joint_self_calibrate(px, x0, bad, cpar, max_nfev=150, free_radial=())
    after = np.nanmedian(loco_residuals(px, cpar, res.cals, vpar))
    assert before > 0.3
    assert after < 0.4 * before
    assert res.rms_after_px < res.rms_before_px


def test_inputs_are_not_modified(rig):
    cpar, _vpar, cals, _pts, px = rig
    bad = _perturbed(cals)
    xh = [c.int_par.xh for c in bad]
    joint_self_calibrate(px[:50], np.zeros((50, 3)) + 1.0, bad, cpar, max_nfev=2)
    assert [c.int_par.xh for c in bad] == xh


def _scaled_about_cam0(cals, s):
    """Cameras as a calibration with a scale error: everything stretched by ``s`` about camera 1 (tracers cannot see it)."""
    out = [copy.deepcopy(c) for c in cals]
    c0 = np.asarray(cals[0].get_pos(), float)
    for c, orig in zip(out[1:], cals[1:]):
        c.set_pos(c0 + s * (np.asarray(orig.get_pos(), float) - c0))
    return out


def _starts(px, cals, cpar, vpar):
    from openptv2.selfcal import triangulate_pixels

    return triangulate_pixels(px, cals, cpar, vpar)


def _dumbbell_obs(cals, cpar, length, k=60, seed=5):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-8, 8, (k * 4, 3))
    d = rng.normal(size=(k * 4, 3))
    b = a + length * d / np.linalg.norm(d, axis=1)[:, None]
    pa = np.stack([project_pixels(a, c, cpar) for c in cals], axis=1)
    pb = np.stack([project_pixels(b, c, cpar) for c in cals], axis=1)
    ok = np.all(np.stack([(p[:, :, 0] > 5) & (p[:, :, 0] < cpar.imx - 5) & (p[:, :, 1] > 5) & (p[:, :, 1] < cpar.imy - 5)
                          for p in (pa, pb)]), axis=(0, 2))
    return np.stack([pa[ok][:k], pb[ok][:k]], axis=1)


def _mean_length(cals, cpar, vpar, obs):
    k = len(obs)
    xyz = _starts(obs.reshape(2 * k, 4, 2), cals, cpar, vpar).reshape(k, 2, 3)
    return float(np.mean(np.linalg.norm(xyz[:, 0] - xyz[:, 1], axis=1)))


def test_dumbbell_sets_the_scale_that_tracers_cannot(rig):
    from openptv2.selfcal import Dumbbell

    cpar, vpar, cals, _pts, px = rig
    length = 5.0
    obs = _dumbbell_obs(cals, cpar, length)
    bad = _scaled_about_cam0(cals, 1.02)
    x0 = _starts(px, bad, cpar, vpar)
    # tracers alone keep the wrong scale ...
    free = joint_self_calibrate(px, x0, bad, cpar, max_nfev=120, free_pp=False, free_radial=(), free_decentering=False,
                                ext_sigma_scale=200)
    assert abs(_mean_length(free.cals, cpar, vpar, obs) / length - 1) > 0.01
    # ... a dumbbell of known length brings the metric scale back
    anchored = joint_self_calibrate(px, x0, bad, cpar, max_nfev=200, free_pp=False, free_radial=(),
                                    free_decentering=False, ext_sigma_scale=200, dumbbell=Dumbbell(obs, length), vpar=vpar)
    assert abs(_mean_length(anchored.cals, cpar, vpar, obs) / length - 1) < 0.003
    assert anchored.length_rms_mm_after < 0.3 * anchored.length_rms_mm_before
    for c, ref, c_bad in zip(anchored.cals[1:], cals[1:], bad[1:]):
        before = np.linalg.norm(np.asarray(c_bad.get_pos()) - np.asarray(ref.get_pos()))
        after = np.linalg.norm(np.asarray(c.get_pos()) - np.asarray(ref.get_pos()))
        assert after < 0.1 * before


def test_known_body_fixes_scale_and_frame(rig):
    from openptv2.selfcal import KnownBody

    cpar, vpar, cals, pts, px = rig
    bad = _scaled_about_cam0(cals, 1.003)
    x0 = _starts(px, bad, cpar, vpar)
    body = KnownBody(pts[:40], px[:40])                  # points with known world coordinates, seen by the true cameras
    res = joint_self_calibrate(px[40:], x0[40:], bad, cpar, hold_cam=-1, max_nfev=200, free_pp=False, free_radial=(),
                               free_decentering=False, ext_sigma_scale=200, body=body)
    assert res.body_rms_mm_after < 0.2
    start = _scaled_about_cam0(cals, 1.003)
    for c, ref, c_bad in zip(res.cals, cals, start):
        before = np.linalg.norm(np.asarray(c_bad.get_pos()) - np.asarray(ref.get_pos()))
        after = np.linalg.norm(np.asarray(c.get_pos()) - np.asarray(ref.get_pos()))
        assert after <= max(0.1 * before, 0.6)          # mm; 40 body points, errors started at up to 3.5 mm


def test_dumbbell_from_store_reads_frames_with_two_targets(tmp_path):
    from openptv2.algorithms.tracking_frame_buf import TargetArray
    from openptv2.selfcal import dumbbell_from_store
    from openptv2.storage import RunStore

    st = RunStore(tmp_path / "run.zarr", mode="a")
    for frame in (1, 2, 3):
        for cam in range(2):
            n = 2 if frame != 2 else 3                      # frame 2 has a stray third target: skipped
            targs = TargetArray(n)
            for i in range(n):
                targs[i].set_pnr(i)
                targs[i].set_pos([10.0 * frame + cam, 20.0 + 5 * i])
                targs[i].set_pixel_counts(5, 2, 2)
                targs[i].set_sum_grey_value(100)
                targs[i].set_tnr(0)
            st.write_targets(cam, frame, targs)
    d = dumbbell_from_store(st, 2, 25.0, first=1, last=3)
    assert d.obs_px.shape == (2, 2, 2, 2)                    # frames 1 and 3, (end, camera, xy)
    assert d.length_mm == 25.0
    assert np.allclose(d.obs_px[0, 0, :, 0], [10.0, 11.0])
