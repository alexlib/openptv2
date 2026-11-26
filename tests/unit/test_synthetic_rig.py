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
