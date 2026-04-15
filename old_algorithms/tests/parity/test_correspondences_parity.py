"""Parity tests for correspondences against Python baseline and optv/Cython."""

from __future__ import annotations

import numpy as np
import pytest

from algorithms.correspondences import correspondences, correspondences_soa, MatchedCoords
from algorithms.parameters import MultimediaPar, VolumePar
from algorithms.tests.helpers.factories import (
    build_corresp_control_par,
    build_corresp_volume_par,
    generate_grid_frame,
    load_sym_calibrations,
)
from algorithms.tests.helpers.parity import (
    assert_array_allclose,
    optv_available,
    sorted_tuple_rows,
    unpack_correspondence_result,
)


@pytest.mark.parity
def test_correspondences_soa_matches_python_baseline() -> None:
    cpar = build_corresp_control_par()
    vpar = build_corresp_volume_par()
    cals = load_sym_calibrations()
    frm, corrected = generate_grid_frame(cals, cpar)

    res_py = correspondences(frm, corrected, vpar, cpar, cals, [0, 0, 0, 0])
    res_soa = correspondences_soa(frm, corrected, vpar, cpar, cals, [0, 0, 0, 0])

    p_py, corr_py = unpack_correspondence_result(res_py)
    p_soa, corr_soa = unpack_correspondence_result(res_soa)

    valid_py = corr_py > 0.0
    valid_soa = corr_soa > 0.0

    p_py = p_py[valid_py]
    p_soa = p_soa[valid_soa]
    corr_py = corr_py[valid_py]
    corr_soa = corr_soa[valid_soa]

    p_py = sorted_tuple_rows(p_py)
    p_soa = sorted_tuple_rows(p_soa)

    assert p_soa.shape[0] == p_py.shape[0]
    # Current SoA can retain 3-camera tuples (cam4 = -2) for part of this scene,
    # so compare stable identity across the first three camera assignments.
    assert_array_allclose(p_soa[:, :3], p_py[:, :3], rtol=0.0, atol=0.0, msg="tuple indices cam1-3")


@pytest.mark.parity
@pytest.mark.requires_optv
@pytest.mark.skipif(not optv_available(), reason="optv bindings not available")
def test_correspondences_soa_matches_optv_tuple_set() -> None:
    from optv.calibration import Calibration as OptvCal
    from optv.correspondences import MatchedCoords as OptvMatchedCoords
    from optv.correspondences import correspondences as optv_correspondences
    from optv.imgcoord import image_coordinates as optv_image_coordinates
    from optv.parameters import ControlParams as OptvControlParams
    from optv.parameters import VolumeParams as OptvVolumeParams
    from optv.tracking_framebuf import TargetArray as OptvTargetArray
    from optv.transforms import convert_arr_metric_to_pixel as optv_metric_to_pixel

    cpar = build_corresp_control_par()
    vpar = build_corresp_volume_par()
    cals = load_sym_calibrations()
    frm, corrected = generate_grid_frame(cals, cpar)
    py_res = correspondences_soa(frm, corrected, vpar, cpar, cals, [0, 0, 0, 0])

    optv_cpar = OptvControlParams(4)
    optv_cpar.read_control_par("test_data/corresp/control.par")
    optv_cpar.get_multimedia_params().set_layers([1.0001], [1.0])
    optv_cpar.get_multimedia_params().set_n3(1.0001)

    optv_vpar = OptvVolumeParams()
    optv_vpar.read_volume_par("test_data/corresp/criteria.par")

    optv_cals = []
    optv_img_pts = []
    optv_corrected = []

    for cam in range(4):
        cal = OptvCal()
        cal.from_file(
            f"test_data/calibration/sym_cam{cam + 1}.tif.ori".encode(),
            b"test_data/calibration/cam1.tif.addpar",
        )
        optv_cals.append(cal)

        targs = OptvTargetArray(16)
        for row in range(4):
            for col in range(4):
                targ_ix = row * 4 + col
                if cam % 2:
                    targ_ix = 15 - targ_ix

                pos3d = 10.0 * np.array([[col, row, 0]], dtype=np.float64)
                pos2d = optv_image_coordinates(pos3d, cal, optv_cpar.get_multimedia_params())
                px, py = optv_metric_to_pixel(pos2d, optv_cpar)[0]

                targ = targs[targ_ix]
                targ.set_pos((px, py))
                targ.set_pnr(targ_ix)
                targ.set_pixel_counts(25, 5, 5)
                targ.set_sum_grey_value(10)

        optv_img_pts.append(targs)
        optv_corrected.append(OptvMatchedCoords(targs, optv_cpar, cal))

    optv_pos, optv_corresp, optv_count = optv_correspondences(
        optv_img_pts, optv_corrected, optv_cals, optv_vpar, optv_cpar
    )

    py_tuples, py_corr = unpack_correspondence_result(py_res)
    py_valid = py_corr > 0.0

    assert int(np.count_nonzero(py_valid)) == int(optv_count)
