"""Leave-one-camera-out consistency (openptv2.loco) on the small cavity rig."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from openptv2.loco import loco_residuals, pixels_to_flat, project_pixels, summarize
from openptv2.recorrespond import load_parameters

pytestmark = pytest.mark.unit

YAML = Path(__file__).resolve().parents[2] / "test_data" / "test_cavity_small" / "parameters_Run1.yaml"


@pytest.fixture(scope="module")
def rig():
    cpar, vpar, cals, _ = load_parameters(YAML)
    rng = np.random.default_rng(1)
    pts = rng.uniform(-10, 10, (3000, 3))
    px = np.stack([project_pixels(pts, c, cpar) for c in cals], axis=1)  # (N, 4, 2)
    ok = np.all((px[:, :, 0] > 5) & (px[:, :, 0] < cpar.imx - 5) & (px[:, :, 1] > 5) & (px[:, :, 1] < cpar.imy - 5), axis=1)
    assert ok.sum() > 100
    return cpar, vpar, cals, pts[ok][:300], px[ok][:300]


def test_exact_observations_have_tiny_residuals(rig):
    cpar, vpar, cals, _pts, px = rig
    res = loco_residuals(px, cpar, cals, vpar)
    assert res.shape == (len(px), 4)
    assert np.nanmax(res) < 0.05


def test_pixel_round_trip_of_flat_conversion_is_finite(rig):
    cpar, _vpar, cals, _pts, px = rig
    flat = pixels_to_flat(px[:, 0], cals[0], cpar)
    assert flat.shape == (len(px), 2) and np.all(np.isfinite(flat))


def test_a_shifted_observation_shows_in_its_own_camera(rig):
    cpar, vpar, cals, _pts, px = rig
    bad = px.copy()
    bad[:, 2, 0] += 1.5  # a merged blob: camera 3 sees the particle 1.5 px off
    res = loco_residuals(bad, cpar, cals, vpar)
    assert np.median(res[:, 2]) > 1.0
    assert np.median(res[:, 2]) > 5 * np.median(res[:, [0, 1, 3]])


def test_unused_camera_and_too_few_cameras_are_nan(rig):
    cpar, vpar, cals, _pts, px = rig
    p = px[:50].copy()
    p[:10, 1] = np.nan          # camera 2 unused for the first 10 points: three cameras left
    p[10:20, 1:3] = np.nan      # two cameras left for points 10-19
    res = loco_residuals(p, cpar, cals, vpar)            # default: three other cameras required
    assert np.all(np.isnan(res[:10, 1]))                 # unused camera has no residual
    assert np.all(np.isnan(res[:20]))                    # holding one out leaves < 3 others
    assert np.all(np.isfinite(res[20:]))
    loose = loco_residuals(p, cpar, cals, vpar, min_other=2)
    assert np.all(np.isfinite(loose[:10, [0, 2, 3]]))    # two other cameras are enough when asked
    assert np.all(np.isnan(loose[10:20]))                # one other camera never is
    s = summarize(res)
    assert s["n"] == 4 * 30 and s["median_px"] < 0.05
    assert summarize(np.full((2, 4), np.nan)) == {"n": 0}
