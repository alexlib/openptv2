"""Coverage-focused unit tests for openptv2.algorithms.sortgrid.

Small, fast tests exercising the branches of sortgrid.py. Reference values
reused from tests/unit/test_sortgrid.py where possible.
"""

import numpy as np

from openptv2.algorithms.sortgrid import (
    sortgrid,
    nearest_neighbour_pix,
    _nearest_neighbour_arr,
    read_sortgrid_par,
    read_calblock,
    is_compiled,
)
from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.parameters import ControlPar
from openptv2.algorithms.tracking_frame_buf import Target, read_targets


# --- nearest_neighbour_pix ---


def test_nearest_neighbour_pix_negative_eps():
    targets = [Target(x=1.0, y=1.0)]
    assert nearest_neighbour_pix(targets, 1.0, 1.0, -1.0) == -999


def test_nearest_neighbour_pix_no_match():
    targets = [Target(x=1000.0, y=1000.0)]
    assert nearest_neighbour_pix(targets, 0.0, 0.0, 1.0) == -999


def test_nearest_neighbour_pix_picks_closest():
    targets = [Target(x=10.0, y=10.0), Target(x=12.0, y=10.0)]
    assert nearest_neighbour_pix(targets, 11.4, 10.0, 5.0) == 1
    assert nearest_neighbour_pix(targets, 10.4, 10.0, 5.0) == 0


# --- _nearest_neighbour_arr ---


def test_nearest_neighbour_arr_negative_eps():
    px = np.array([1.0])
    py = np.array([1.0])
    assert _nearest_neighbour_arr(px, py, 1.0, 1.0, -1.0) == -999


def test_nearest_neighbour_arr_no_match():
    px = np.array([100.0, 200.0])
    py = np.array([100.0, 200.0])
    assert _nearest_neighbour_arr(px, py, 0.0, 0.0, 1.0) == -999


def test_nearest_neighbour_arr_picks_closest():
    px = np.array([10.0, 12.0, 20.0])
    py = np.array([10.0, 10.0, 20.0])
    assert _nearest_neighbour_arr(px, py, 11.4, 10.0, 5.0) == 1
    assert _nearest_neighbour_arr(px, py, 10.4, 10.0, 5.0) == 0


# --- read_sortgrid_par ---


def test_read_sortgrid_par_valid():
    assert read_sortgrid_par("test_data/parameters/sortgrid.par") == 25


def test_read_sortgrid_par_missing(tmp_path):
    assert read_sortgrid_par(tmp_path / "nope.par") == 0


def test_read_sortgrid_par_bad_content(tmp_path):
    p = tmp_path / "bad.par"
    p.write_text("not-a-number")
    assert read_sortgrid_par(p) == 0


# --- read_calblock ---


# happy-path read_calblock is covered by tests/unit/test_sortgrid.py;
# only the edge cases (missing/empty) are exercised here.


def test_read_calblock_missing(tmp_path):
    fix, num_points = read_calblock(tmp_path / "nope.txt")
    assert num_points == 0
    assert fix.shape == (0, 3)


def test_read_calblock_empty(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("")
    fix, num_points = read_calblock(p)
    assert num_points == 0
    assert fix.shape == (0, 3)


# --- sortgrid ---


def _load_fixtures():
    cal = Calibration.from_file(
        "test_data/calibration/cam1.tif.ori",
        "test_data/calibration/cam1.tif.addpar",
    )
    cpar = ControlPar.from_yaml("test_data/parameters.yaml")
    fix, nfix = read_calblock("test_data/calibration/calblock.txt")
    return cal, cpar, fix, nfix


def _patch_metric_to_pixel(monkeypatch, px, py):
    """Force metric_to_pixel (late-imported inside sortgrid) to a fixed result.

    sortgrid's pure-Python dependency trafo.metric_to_pixel raises
    UnboundLocalError in interpreted mode, so we stub it to exercise the
    sortgrid loop body deterministically.
    """
    import openptv2.algorithms.trafo as trafo

    monkeypatch.setattr(trafo, "metric_to_pixel", lambda *a, **k: (px, py))


def test_sortgrid_small_match(monkeypatch):
    """Small (non-vectorized) list: projected point lands on a target."""
    cal, cpar, fix, nfix = _load_fixtures()
    _patch_metric_to_pixel(monkeypatch, 100.0, 100.0)
    pix = [Target(pnr=7, x=100.0, y=100.0), Target(pnr=8, x=500.0, y=500.0)]
    sorted_pix = sortgrid(cal, cpar, nfix, fix, len(pix), 25, pix)
    assert len(sorted_pix) == nfix
    # Every reference point projects to (100,100) -> matches target 0.
    assert all(t.pnr == i and t.x == 100.0 for i, t in enumerate(sorted_pix))


def test_sortgrid_out_of_bounds(monkeypatch):
    """Projected point outside the image frame -> entry left unmatched."""
    cal, cpar, fix, nfix = _load_fixtures()
    _patch_metric_to_pixel(monkeypatch, -1000.0, -1000.0)
    pix = [Target(pnr=0, x=100.0, y=100.0)]
    sorted_pix = sortgrid(cal, cpar, nfix, fix, len(pix), 25, pix)
    assert all(t.pnr == -999 for t in sorted_pix)


def test_sortgrid_no_neighbour(monkeypatch):
    """In-bounds projection but no target within eps -> unmatched."""
    cal, cpar, fix, nfix = _load_fixtures()
    _patch_metric_to_pixel(monkeypatch, 100.0, 100.0)
    pix = [Target(pnr=0, x=5000.0, y=5000.0)]
    sorted_pix = sortgrid(cal, cpar, nfix, fix, len(pix), 25, pix)
    assert all(t.pnr == -999 for t in sorted_pix)


def test_sortgrid_vectorized_path(monkeypatch):
    """>=16 targets triggers the vectorized nearest-neighbour branch."""
    cal, cpar, fix, nfix = _load_fixtures()
    _patch_metric_to_pixel(monkeypatch, 100.0, 100.0)
    pix = [Target(pnr=i, x=float(i * 10), y=float(i * 10)) for i in range(16)]
    # target index 10 sits at (100,100) -> exact match
    sorted_pix = sortgrid(cal, cpar, nfix, fix, len(pix), 25, pix)
    assert len(sorted_pix) == nfix
    assert all(t.pnr == i and t.x == 100.0 for i, t in enumerate(sorted_pix))


# --- is_compiled ---


def test_is_compiled_returns_bool():
    assert isinstance(is_compiled(), bool)
