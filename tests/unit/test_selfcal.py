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
