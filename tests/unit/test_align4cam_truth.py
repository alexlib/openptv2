"""Truth test of the 4-camera residual alignment: points with known positions, spoiled processing cameras.

See ``scripts/synthetic_align4cam.py``. The alignment must (1) change nothing when the calibration is right,
(2) remove the jump between 3- and 4-camera positions when it is not, also with noise, outliers, few points and
points outside the fitted region, and (3) never touch the 4-camera positions.
"""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "synthetic_align4cam.py"
CAVITY = Path(__file__).resolve().parents[2] / "test_data" / "test_cavity" / "cal" / "cam1.tif.ori"

pytestmark = pytest.mark.skipif(not CAVITY.exists(), reason="test_cavity calibration not present")


@pytest.fixture(scope="module")
def sc():
    spec = importlib.util.spec_from_file_location("synthetic_align4cam", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_calibration_error_means_no_change(sc):
    r = sc.scenario("A", 1, 1500, kind="truth")
    assert abs(r["err3_al"] - r["err3_raw"]) < 0.1 * r["err3_raw"]
    assert abs(r["jump_al"] - r["jump_raw"]) < 0.1 * r["jump_raw"]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),                                            # calibration error
        dict(scale=3.0),                                   # three times larger
        dict(noise=0.3),                                   # noisier centres
        dict(outliers=0.05),                               # 5 % gross errors among the fit points
        dict(n_fit=150),                                   # a small fit sample
        dict(fit_shell=(0.0, 0.4), eval_shell=(0.4, 1.0)),  # evaluated outside the fitted region
    ],
    ids=["x1", "x3", "noise", "outliers", "few", "extrapolate"],
)
def test_the_three_to_four_camera_jump_collapses(sc, kwargs):
    r = sc.scenario("B", 1, 1500, **kwargs)
    assert r["jump_raw"] > 0.05                            # there is a jump to remove
    assert r["jump_al"] < 0.2 * r["jump_raw"]
    assert r["err3_al"] <= r["err3_raw"] * 1.02            # and the 3-camera error does not grow
