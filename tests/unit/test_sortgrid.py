import random

import numpy as np
import pytest

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.parameters import ControlPar
from openptv2.algorithms.sortgrid import (
    nearest_neighbour_pix,
    read_calblock,
    read_sortgrid_par,
    sortgrid,
)
from openptv2.algorithms.tracking_frame_buf import Target, read_targets


def _has_optv():
    try:
        from optv.orientation import match_detection_to_ref

        return True
    except ImportError:
        return False


# --- Unit tests ---


def test_nearest_neighbour_pix():
    t1 = Target(x=1127.0, y=796.0)
    targets = [t1]

    assert nearest_neighbour_pix(targets, 1128.0, 795.0, 0.0) == -999
    assert nearest_neighbour_pix(targets, 1128.0, 795.0, -1.0) == -999
    assert nearest_neighbour_pix(targets, -1127.0, -796.0, 1e3) == -999
    assert nearest_neighbour_pix(targets, 1127.0, 796.0, 1e-5) == 0


def test_nearest_neighbour_pix_multiple():
    targets = [Target(x=10.0, y=10.0), Target(x=12.0, y=10.0), Target(x=20.0, y=20.0)]
    assert nearest_neighbour_pix(targets, 11.0, 10.0, 5.0) == 0


def test_read_sortgrid_par():
    eps = read_sortgrid_par("test_data/parameters/sortgrid.par")
    assert eps == 25


def test_read_calblock():
    fix, num_points = read_calblock("test_data/calibration/calblock.txt")
    assert num_points == 5
    assert fix.shape == (5, 3)


def test_sortgrid():
    eps = read_sortgrid_par("test_data/parameters/sortgrid.par")
    assert eps == 25

    pix = read_targets("test_data/sample_", 42)
    assert len(pix) == 2

    cal = Calibration.from_file(
        "test_data/calibration/cam1.tif.ori",
        "test_data/calibration/cam1.tif.addpar",
    )
    cpar = ControlPar.from_yaml("test_data/parameters.yaml")
    fix, nfix = read_calblock("test_data/calibration/calblock.txt")
    assert nfix == 5

    sorted_pix = sortgrid(cal, cpar, nfix, fix, len(pix), eps, pix)
    assert len(sorted_pix) == nfix
    assert sorted_pix[0].pnr == -999
    assert sorted_pix[1].pnr == -999

    pix2 = read_targets("test_data/sample_", 42)
    sorted_pix = sortgrid(cal, cpar, nfix, fix, len(pix2), 120, pix2)
    assert sorted_pix[1].pnr == 1
    assert sorted_pix[1].x == 796


def test_sortgrid_does_not_mutate_input():
    """Ensure sortgrid doesn't modify the input target list."""
    pix = read_targets("test_data/sample_", 42)
    original_pnrs = [t.pnr for t in pix]
    original_xs = [t.x for t in pix]

    cal = Calibration.from_file(
        "test_data/calibration/cam1.tif.ori",
        "test_data/calibration/cam1.tif.addpar",
    )
    cpar = ControlPar.from_yaml("test_data/parameters.yaml")
    fix, nfix = read_calblock("test_data/calibration/calblock.txt")

    sortgrid(cal, cpar, nfix, fix, len(pix), 120, pix)

    assert [t.pnr for t in pix] == original_pnrs
    assert [t.x for t in pix] == original_xs


# --- Parity test against Cython bindings ---


@pytest.mark.skipif(not _has_optv(), reason="optv (Cython bindings) not available")
def test_sortgrid_parity_with_cython():
    """Compare Python sortgrid against C/Cython match_detection_to_ref."""
    from optv.calibration import Calibration as CCalib
    from optv.orientation import match_detection_to_ref
    from optv.parameters import ControlParams as CControlParams
    from optv.tracking_framebuf import TargetArray

    xyz_input = np.array(
        [
            (10, 10, 10),
            (200, 200, 200),
            (600, 800, 100),
            (20, 10, 2000),
            (30, 30, 30),
        ],
        dtype=float,
    )

    # --- C/Cython path ---
    c_cal = CCalib()
    c_cal.from_file(
        "test_data/calibration/cam1.tif.ori",
        "test_data/calibration/cam1.tif.addpar",
    )
    c_cpar = CControlParams(4)
    c_cpar.read_control_par("test_data/control_parameters/control.par")

    from optv.imgcoord import image_coordinates
    from optv.transforms import convert_arr_metric_to_pixel

    xy_metric = image_coordinates(
        xyz_input,
        c_cal,
        c_cpar.get_multimedia_params(),
    )
    xy_pixel = convert_arr_metric_to_pixel(xy_metric, cpar=c_cpar)

    target_array = TargetArray(len(xyz_input))
    for i in range(len(xyz_input)):
        target_array[i].set_pnr(i)
        target_array[i].set_pos((xy_pixel[i][0], xy_pixel[i][1]))

    c_sorted = match_detection_to_ref(
        cal=c_cal,
        ref_pts=xyz_input,
        img_pts=target_array,
        cpar=c_cpar,
    )

    # --- Python path ---
    py_cal = Calibration.from_file(
        "test_data/calibration/cam1.tif.ori",
        "test_data/calibration/cam1.tif.addpar",
    )
    py_cpar = ControlPar.from_file("test_data/control_parameters/control.par")

    py_targets = []
    for i in range(len(xyz_input)):
        py_targets.append(
            Target(
                pnr=i,
                x=xy_pixel[i][0],
                y=xy_pixel[i][1],
            )
        )

    py_sorted = sortgrid(
        py_cal,
        py_cpar,
        len(xyz_input),
        xyz_input,
        len(py_targets),
        25,
        py_targets,
    )

    # Compare
    for i in range(len(xyz_input)):
        c_pnr = c_sorted[i].pnr()
        py_pnr = py_sorted[i].pnr
        assert c_pnr == py_pnr, f"point {i}: C pnr={c_pnr}, Python pnr={py_pnr}"

        if c_pnr != -999:
            c_pos = c_sorted[i].pos()
            assert abs(py_sorted[i].x - c_pos[0]) < 1e-4, (
                f"point {i}: x mismatch C={c_pos[0]} Python={py_sorted[i].x}"
            )
            assert abs(py_sorted[i].y - c_pos[1]) < 1e-4, (
                f"point {i}: y mismatch C={c_pos[1]} Python={py_sorted[i].y}"
            )


@pytest.mark.skipif(not _has_optv(), reason="optv (Cython bindings) not available")
def test_sortgrid_parity_shuffled():
    """Parity test with shuffled targets — mirrors the Cython test pattern."""
    from optv.calibration import Calibration as CCalib
    from optv.imgcoord import image_coordinates
    from optv.orientation import match_detection_to_ref
    from optv.parameters import ControlParams as CControlParams
    from optv.tracking_framebuf import TargetArray
    from optv.transforms import convert_arr_metric_to_pixel

    xyz_input = np.array(
        [
            (10, 10, 10),
            (200, 200, 200),
            (600, 800, 100),
            (20, 10, 2000),
            (30, 30, 30),
        ],
        dtype=float,
    )

    c_cal = CCalib()
    c_cal.from_file(
        "test_data/calibration/cam1.tif.ori",
        "test_data/calibration/cam1.tif.addpar",
    )
    c_cpar = CControlParams(4)
    c_cpar.read_control_par("test_data/control_parameters/control.par")

    xy_metric = image_coordinates(
        xyz_input,
        c_cal,
        c_cpar.get_multimedia_params(),
    )
    xy_pixel = convert_arr_metric_to_pixel(xy_metric, cpar=c_cpar)

    # Shuffle targets
    indices = list(range(len(xyz_input)))
    shuffled = indices[:]
    random.seed(42)
    while shuffled == indices:
        random.shuffle(shuffled)

    # C path — shuffled TargetArray
    c_targets = TargetArray(len(xyz_input))
    for i in range(len(xyz_input)):
        c_targets[shuffled[i]].set_pos((xy_pixel[i][0], xy_pixel[i][1]))
        c_targets[shuffled[i]].set_pnr(i)

    c_sorted = match_detection_to_ref(
        cal=c_cal,
        ref_pts=xyz_input,
        img_pts=c_targets,
        cpar=c_cpar,
    )

    # Python path — shuffled list
    py_cal = Calibration.from_file(
        "test_data/calibration/cam1.tif.ori",
        "test_data/calibration/cam1.tif.addpar",
    )
    py_cpar = ControlPar.from_file("test_data/control_parameters/control.par")

    py_targets = [Target() for _ in range(len(xyz_input))]
    for i in range(len(xyz_input)):
        py_targets[shuffled[i]] = Target(
            pnr=i,
            x=xy_pixel[i][0],
            y=xy_pixel[i][1],
        )

    py_sorted = sortgrid(
        py_cal,
        py_cpar,
        len(xyz_input),
        xyz_input,
        len(py_targets),
        25,
        py_targets,
    )

    for i in range(len(xyz_input)):
        c_pnr = c_sorted[i].pnr()
        py_pnr = py_sorted[i].pnr
        assert c_pnr == py_pnr, f"point {i}: C pnr={c_pnr}, Python pnr={py_pnr}"
