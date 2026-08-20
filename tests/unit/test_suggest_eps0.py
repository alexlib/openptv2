"""_pick_eps0 chooses the knee: largest spurious-free eps0 at max correct.

Also an integration test of suggest_eps0 on the test_data/test_splitter fixture
(4-camera splitter with a committed calblock + detected cal targets).
"""

from pathlib import Path

import pytest

from openptv2.autocalibration import (
    _load_dataset_params,
    _pick_eps0,
    cam_files,
    resolve_calblock,
    suggest_eps0,
)

SPLITTER = Path("test_data/test_splitter")


@pytest.mark.unit
def test_pick_knee_from_aorta_shape():
    # Real aorta sweep shape: correct saturates at 34, first wrong at 0.099.
    rows = [
        {"eps0": 0.012, "correct": 2, "wrong": 0},
        {"eps0": 0.045, "correct": 23, "wrong": 0},
        {"eps0": 0.058, "correct": 31, "wrong": 0},
        {"eps0": 0.076, "correct": 34, "wrong": 0},  # <- knee
        {"eps0": 0.099, "correct": 34, "wrong": 2},
        {"eps0": 0.480, "correct": 34, "wrong": 20},
    ]
    eps0, correct, wrong = _pick_eps0(rows)
    assert eps0 == 0.076 and correct == 34 and wrong == 0


@pytest.mark.unit
def test_pick_largest_clean_when_correct_still_rising():
    # If correct keeps rising while clean, take the largest clean (most matches).
    rows = [
        {"eps0": 0.02, "correct": 5, "wrong": 0},
        {"eps0": 0.05, "correct": 12, "wrong": 0},
        {"eps0": 0.10, "correct": 20, "wrong": 0},  # <- largest clean, max correct
        {"eps0": 0.20, "correct": 20, "wrong": 4},
    ]
    assert _pick_eps0(rows)[0] == 0.10


@pytest.mark.unit
def test_pick_falls_back_when_never_clean():
    # No spurious-free row -> maximize correct - 2*wrong.
    rows = [
        {"eps0": 0.05, "correct": 10, "wrong": 1},  # 10 - 2 = 8
        {"eps0": 0.10, "correct": 15, "wrong": 2},  # 15 - 4 = 11 <- best
        {"eps0": 0.20, "correct": 16, "wrong": 6},  # 16 - 12 = 4
    ]
    assert _pick_eps0(rows)[0] == 0.10


@pytest.mark.unit
def test_pick_empty():
    assert _pick_eps0([]) is None


@pytest.mark.unit
def test_suggest_eps0_on_test_splitter():
    """End-to-end on the 4-camera splitter fixture (committed cal targets):
    the recommended eps0 must be a real spurious-free knee, and widening the
    band must eventually introduce spurious quadruplets."""
    if not (SPLITTER / "cal" / "cam_1.tif_targets").exists():
        pytest.skip("test_splitter cal targets not present")
    from openptv2.algorithms.calibration import Calibration

    base = SPLITTER.resolve()
    cpar = _load_dataset_params(base, resolve_calblock(base)).cpar
    cals = [
        Calibration.from_file(*[str(p) for p in cam_files(base, c)[1:]])
        for c in range(cpar.num_cams)
    ]
    r = suggest_eps0(base, cpar, cals)
    assert r is not None
    assert r["recommended"] is not None and r["recommended"] > 0
    assert r["max_correct"] >= 1
    # the recommended eps0 is a real sweep row and is spurious-free at max correct
    row = next(x for x in r["sweep"] if x["eps0"] == r["recommended"])
    assert row["wrong"] == 0
    assert row["correct"] == r["max_correct"]
    # sanity: a wide band eventually produces spurious quadruplets
    assert any(x["wrong"] > 0 for x in r["sweep"])


@pytest.mark.unit
def test_suggest_eps0_non_four_camera_returns_none():
    class _Cpar:
        num_cams = 3

    assert suggest_eps0(SPLITTER.resolve(), _Cpar(), []) is None
