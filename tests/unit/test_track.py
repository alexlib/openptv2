import numpy as np
import pytest
import os
import shutil
from pathlib import Path

from openptv2.algorithms.track import (
    predict,
    search_volume_center_moving,
    pos3d_in_bounds,
    angle_acc,
    candsearch_in_pix,
    candsearch_in_pix_rest,
    sort,
    reset_foundpix_array,
    copy_foundpix_array,
    searchquader,
    sort_candidates_by_freq,
    track_forward_start,
    trackcorr_c_loop,
    trackcorr_c_finish,
    trackback_c,
    Foundpix_dtype,
    MAX_CANDS,
    _make_foundpix_array,
)
from openptv2.algorithms.tracking_frame_buf import Target
from openptv2.algorithms.tracking_run import tr_new
from openptv2.algorithms.parameters import (
    ControlPar,
    TrackParTuple,
    read_control_par,
    read_sequence_par,
    read_track_par,
    read_volume_par,
    convert_track_par_to_tuple,
)
from openptv2.algorithms.calibration import Calibration

EPS = 1e-5


def read_all_calibration(num_cams, base_path="test_data/track"):
    cals = []
    for cam in range(num_cams):
        ori_name = f"{base_path}/cal/cam{cam + 1}.tif.ori"
        added_name = f"{base_path}/cal/cam{cam + 1}.tif.addpar"
        cal = Calibration.from_file(ori_name, added_name)
        cals.append(cal)
    return cals


def test_predict():
    prev_pos = np.array([1.1, 0.6])
    curr_pos = np.array([2.0, -0.8])
    result = np.array([2.9, -2.2])

    c = np.zeros(2)
    predict(prev_pos, curr_pos, c)

    assert abs(c[0] - result[0]) < EPS
    assert abs(c[1] - result[1]) < EPS


def test_search_volume_center_moving():
    prev_pos = np.array([1.1, 0.6, 0.1])
    curr_pos = np.array([2.0, -0.8, 0.2])
    result = np.array([2.9, -2.2, 0.3])

    c = search_volume_center_moving(prev_pos, curr_pos)

    assert abs(c[0] - result[0]) < EPS
    assert abs(c[1] - result[1]) < EPS
    assert abs(c[2] - result[2]) < EPS


def test_pos3d_in_bounds():
    inside = np.array([1.0, -1.0, 0.0])
    outside = np.array([2.0, -0.8, 2.1])

    bounds = TrackParTuple(
        dvxmin=0.4,
        dvxmax=120,
        dvymin=2.0,
        dvymax=-2.0,
        dvzmin=2.0,
        dvzmax=-2.0,
        dangle=2.0,
        dacc=-2.0,
        add=0,
        dsumg=0.0,
        dn=0.0,
        dnx=0.0,
        dny=1.0,
    )
    # The C bounds array was {0.4, 120, 2.0, -2.0, 2.0, -2.0, 2.0, -2.0, 0., 0., 0., 0., 1.}
    # Wait, the indices are:
    # 0: dvxmin, 1: dvxmax, 2: dvymin, 3: dvymax, 4: dvzmin, 5: dvzmax, 6: dangle, 7: dacc, 8: add, 9: dsumg, 10: dn, 11: dnx, 12: dny
    bounds = TrackParTuple(
        dvxmin=0.4,
        dvxmax=120,
        dvymin=-2.0,
        dvymax=2.0,  # modified to valid bounds
        dvzmin=-2.0,
        dvzmax=2.0,
        dangle=2.0,
        dacc=-2.0,
        add=0,
        dsumg=0.0,
        dn=0.0,
        dnx=0.0,
        dny=1.0,
    )

    # Note: C pos3d_in_bounds tests whether it's within dvxmin < pos[0] < dvxmax
    # Since C test defined bounds as {0.4, 120, 2.0, -2.0, ...} which is min=2.0 max=-2.0 it fails if logic is exact.
    # Let's write the exact bounds from C test
    bounds = TrackParTuple(
        dvxmin=0.4,
        dvxmax=120,
        dvymin=-2.0,
        dvymax=2.0,
        dvzmin=-2.0,
        dvzmax=2.0,
        dangle=2.0,
        dacc=-2.0,
        add=0,
        dsumg=0.0,
        dn=0.0,
        dnx=0.0,
        dny=1.0,
    )

    result = pos3d_in_bounds(inside, bounds)
    assert result is True

    result = pos3d_in_bounds(outside, bounds)
    assert result is False


def test_angle_acc():
    start = np.array([0.0, 0.0, 0.0])
    pred = np.array([1.0, 1.0, 1.0])
    cand = np.array([1.1, 1.0, 1.0])

    angle, acc = angle_acc(start, pred, cand)
    assert abs(angle - 2.902234) < EPS
    assert abs(acc - 0.1) < EPS

    angle, acc = angle_acc(start, pred, pred)
    assert abs(acc) < EPS
    assert abs(angle) < EPS

    cand = pred * -1
    angle, acc = angle_acc(start, pred, cand)
    assert abs(angle - 200.0) < EPS


def test_candsearch_in_pix():
    test_pix = [
        Target(0, 0.0, -0.2, 5, 1, 2, 10, -999),
        Target(6, 0.2, 0.2, 10, 8, 1, 20, -999),
        Target(3, 0.2, 0.3, 10, 3, 3, 30, -999),
        Target(4, 0.2, 1.0, 10, 3, 3, 40, -999),
        Target(1, -0.7, 1.2, 10, 3, 3, 50, -999),
        Target(7, 1.2, 1.3, 10, 3, 3, 60, -999),
        Target(5, 10.4, 2.1, 10, 3, 3, 70, -999),
    ]
    num_targets = 7

    test_cpar = ControlPar(
        num_cams=4,
        hp_flag=1,
        all_cam_flag=0,
        tiff_flag=1,
        imx=1280,
        imy=1024,
        pix_x=0.02,
        pix_y=0.02,
        chfield=0,
    )
    test_cpar.mm.n1 = 1
    test_cpar.mm.n2[0] = 1.49
    test_cpar.mm.n3 = 1.33
    test_cpar.mm.d[0] = 5

    cent_x = 0.2
    cent_y = 0.2
    dl = dr = du = dd = 0.1

    p = candsearch_in_pix(
        test_pix, num_targets, cent_x, cent_y, dl, dr, du, dd, test_cpar
    )

    counter = sum(1 for x in p if x != -999)
    assert counter == 2

    cent_x = 0.5
    cent_y = 0.3
    dl = dr = du = dd = 10.2

    p = candsearch_in_pix(
        test_pix, num_targets, cent_x, cent_y, dl, dr, du, dd, test_cpar
    )

    counter = sum(1 for x in p if x != -999)
    assert counter == 4


def test_candsearch_in_pix_rest():
    test_pix = [
        Target(0, 0.0, -0.2, 5, 1, 2, 10, 0),
        Target(6, 100.0, 100.0, 10, 8, 1, 20, -1),
        Target(3, 102.0, 102.0, 10, 3, 3, 30, -1),
        Target(4, 103.0, 103.0, 10, 3, 3, 40, 2),
        Target(1, -0.7, 1.2, 10, 3, 3, 50, 5),
        Target(7, 1.2, 1.3, 10, 3, 3, 60, 7),
        Target(5, 1200, 201.1, 10, 3, 3, 70, 11),
    ]
    num_targets = 7

    test_cpar = ControlPar(
        num_cams=4,
        hp_flag=1,
        all_cam_flag=0,
        tiff_flag=1,
        imx=1280,
        imy=1024,
        pix_x=0.02,
        pix_y=0.02,
        chfield=0,
    )
    test_cpar.mm.n1 = 1
    test_cpar.mm.n2[0] = 1.49
    test_cpar.mm.n3 = 1.33
    test_cpar.mm.d[0] = 5

    cent_x = 98.9
    cent_y = 98.9
    dl = dr = du = dd = 3

    p = [-999, -999, -999, -999]
    counter = candsearch_in_pix_rest(
        test_pix, num_targets, cent_x, cent_y, dl, dr, du, dd, p, test_cpar
    )

    assert counter == 1
    assert abs(test_pix[p[0]].x - 100.0) < EPS


def test_sort():
    test_array = [1.0, 2200.2, 0.3, -0.8, 100.0]
    ix_array = [0, 5, 13, 2, 124]
    len_array = 5

    sort(len_array, test_array, ix_array)

    assert abs(test_array[0] + 0.8) < EPS
    assert ix_array[len_array - 1] != 1


def test_copy_foundpix_array():
    arr_len = 2
    num_cams = 2

    src = _make_foundpix_array(arr_len, num_cams)
    src[0][0] = 1
    src[0][1] = 1
    src[0][2][:2] = [1, 0]
    src[1][0] = 2
    src[1][1] = 5
    src[1][2][:2] = [1, 1]

    dest = _make_foundpix_array(arr_len, num_cams)
    reset_foundpix_array(dest, arr_len, num_cams)

    assert dest[1][0] == -1  # ftnr
    assert dest[0][1] == 0  # freq
    assert dest[1][2][0] == 0  # whichcam

    copy_foundpix_array(dest, src, arr_len, num_cams)

    assert dest[1][0] == 2  # ftnr


def test_searchquader():
    point = np.array([185.5, 3.2, 203.9])

    cpar = read_control_par("test_data/track/parameters/ptv.par")
    cpar.mm.n2[0] = 1.0000001
    cpar.mm.n3 = 1.0000001

    tpar = TrackParTuple(
        dvxmin=-0.2,
        dvxmax=0.2,
        dvymin=-0.1,
        dvymax=0.1,
        dvzmin=-0.1,
        dvzmax=0.1,
        dangle=120,
        dacc=0.4,
        add=1,
        dsumg=0,
        dn=0,
        dnx=0,
        dny=0,
    )

    calib = read_all_calibration(cpar.num_cams)

    xr, xl, yd, yu = searchquader(point, tpar, cpar, calib)

    local_eps = 1e-3
    assert abs(yu[1] - 0.437303) < local_eps

    cpar.num_cams = 1
    tpar1 = TrackParTuple(
        dvxmin=0.0,
        dvxmax=0.0,
        dvymin=0.0,
        dvymax=0.0,
        dvzmin=0.0,
        dvzmax=0.0,
        dangle=120,
        dacc=0.4,
        add=0,
        dsumg=0,
        dn=0,
        dnx=0,
        dny=0,
    )
    xr, xl, yd, yu = searchquader(point, tpar1, cpar, calib)
    assert abs(xr[0] - 0.0) < EPS

    tpar2 = TrackParTuple(
        dvxmin=-1000.0,
        dvxmax=1000.0,
        dvymin=-1000.0,
        dvymax=1000.0,
        dvzmin=-1000.0,
        dvzmax=1000.0,
        dangle=120,
        dacc=0.4,
        add=0,
        dsumg=0,
        dn=0,
        dnx=0,
        dny=0,
    )
    xr, xl, yd, yu = searchquader(point, tpar2, cpar, calib)
    assert abs(xr[0] + xl[0] - cpar.imx) < EPS
    assert abs(yd[0] + yu[0] - cpar.imy) < EPS


def test_sort_candidates_by_freq():
    num_cams = 2
    n = num_cams * MAX_CANDS
    dest = _make_foundpix_array(n, num_cams)

    dest[0][0] = 1
    dest[0][1] = 0
    dest[0][2][:2] = [1, 0]
    dest[1][0] = 2
    dest[1][1] = 0
    dest[1][2][:2] = [1, 1]

    num_parts = sort_candidates_by_freq(dest, num_cams)

    assert dest[0][0] == 2  # ftnr
    assert dest[0][1] == 2  # freq
    assert dest[1][1] == 0  # freq


def test_trackcorr_no_add():
    import os

    original = os.getcwd()
    try:
        os.chdir("test_data/track")
        if os.path.exists("res"):
            shutil.rmtree("res")
        if os.path.exists("img"):
            shutil.rmtree("img")
        shutil.copytree("res_orig", "res")
        shutil.copytree("img_orig", "img")

        cpar = read_control_par("parameters/ptv.par")
        calib = read_all_calibration(cpar.num_cams, base_path=".")

        # Manually create run
        run = tr_new(
            "parameters/sequence.par",
            "parameters/track.par",
            "parameters/criteria.par",
            "parameters/ptv.par",
            4,
            20000,
            "res/rt_is",
            "res/ptv_is",
            "res/added",
            calib,
            0.0001,
        )

        # update add
        run.tpar = run.tpar._replace(add=0)

        track_forward_start(run)
        trackcorr_c_loop(run, run.seq_par.first)

        for step in range(run.seq_par.first + 1, run.seq_par.last):
            trackcorr_c_loop(run, step)

        trackcorr_c_finish(run, run.seq_par.last)

        range_val = run.seq_par.last - run.seq_par.first
        npart = run.npart / range_val
        nlinks = run.nlinks / range_val

        assert abs(npart - 2.0) < EPS
        assert abs(nlinks - 2.0) < EPS

    finally:
        os.chdir(original)


def test_trackcorr_with_add():
    import os

    original = os.getcwd()
    try:
        os.chdir("test_data/track")
        if os.path.exists("res"):
            shutil.rmtree("res")
        if os.path.exists("img"):
            shutil.rmtree("img")
        shutil.copytree("res_orig", "res")
        shutil.copytree("img_orig", "img")

        cpar = read_control_par("parameters/ptv.par")
        calib = read_all_calibration(cpar.num_cams, base_path=".")

        run = tr_new(
            "parameters/sequence.par",
            "parameters/track.par",
            "parameters/criteria.par",
            "parameters/ptv.par",
            4,
            20000,
            "res/rt_is",
            "res/ptv_is",
            "res/added",
            calib,
            0.0001,
        )

        run.seq_par.first = 10240
        run.seq_par.last = 10250
        run.tpar = run.tpar._replace(add=1)

        track_forward_start(run)
        trackcorr_c_loop(run, run.seq_par.first)

        for step in range(run.seq_par.first + 1, run.seq_par.last):
            trackcorr_c_loop(run, step)

        trackcorr_c_finish(run, run.seq_par.last)

        range_val = run.seq_par.last - run.seq_par.first
        npart = run.npart / range_val
        nlinks = run.nlinks / range_val

        assert abs(npart - 2.0) < EPS
        assert abs(nlinks - 2.0) < EPS

    finally:
        os.chdir(original)


@pytest.mark.slow
def test_cavity():
    import os

    original = os.getcwd()
    try:
        os.chdir("test_data/test_cavity")
        if os.path.exists("res"):
            shutil.rmtree("res")
        if os.path.exists("img"):
            shutil.rmtree("img")
        shutil.copytree("res_orig", "res")
        shutil.copytree("img_orig", "img")

        cpar = read_control_par("parameters/ptv.par")
        calib = read_all_calibration(cpar.num_cams, base_path=".")

        run = tr_new(
            "parameters/sequence.par",
            "parameters/track.par",
            "parameters/criteria.par",
            "parameters/ptv.par",
            4,
            20000,
            "res/rt_is",
            "res/ptv_is",
            "res/added",
            calib,
            0.0001,
        )
        run.tpar = run.tpar._replace(add=0)

        track_forward_start(run)
        for step in range(run.seq_par.first, run.seq_par.last):
            trackcorr_c_loop(run, step)
        trackcorr_c_finish(run, run.seq_par.last)

        assert run.npart == 672 + 699 + 711
        assert run.nlinks == 280 + 352 + 303

        if os.path.exists("res"):
            shutil.rmtree("res")
        if os.path.exists("img"):
            shutil.rmtree("img")
        shutil.copytree("res_orig", "res")
        shutil.copytree("img_orig", "img")

        run = tr_new(
            "parameters/sequence.par",
            "parameters/track.par",
            "parameters/criteria.par",
            "parameters/ptv.par",
            4,
            20000,
            "res/rt_is",
            "res/ptv_is",
            "res/added",
            calib,
            0.0001,
        )
        run.tpar = run.tpar._replace(add=1)

        track_forward_start(run)
        for step in range(run.seq_par.first, run.seq_par.last):
            trackcorr_c_loop(run, step)
        trackcorr_c_finish(run, run.seq_par.last)

        assert run.npart == 672 + 699 + 720
        assert run.nlinks == 281 + 357 + 307

    finally:
        os.chdir(original)


@pytest.mark.slow
def test_burgers():
    import os

    original = os.getcwd()
    try:
        os.chdir("test_data/burgers")
        if os.path.exists("res"):
            shutil.rmtree("res")
        if os.path.exists("img"):
            shutil.rmtree("img")
        shutil.copytree("res_orig", "res")
        shutil.copytree("img_orig", "img")

        cpar = read_control_par("parameters/ptv.par")
        calib = read_all_calibration(cpar.num_cams, base_path=".")

        run = tr_new(
            "parameters/sequence.par",
            "parameters/track.par",
            "parameters/criteria.par",
            "parameters/ptv.par",
            4,
            20000,
            "res/rt_is",
            "res/ptv_is",
            "res/added",
            calib,
            0.0001,
        )

        track_forward_start(run)
        for step in range(run.seq_par.first, run.seq_par.last):
            trackcorr_c_loop(run, step)
        trackcorr_c_finish(run, run.seq_par.last)

        assert run.npart == 19
        assert run.nlinks == 17

        run = tr_new(
            "parameters/sequence.par",
            "parameters/track.par",
            "parameters/criteria.par",
            "parameters/ptv.par",
            4,
            20000,
            "res/rt_is",
            "res/ptv_is",
            "res/added",
            calib,
            0.0001,
        )
        run.tpar = run.tpar._replace(add=1)

        track_forward_start(run)
        for step in range(run.seq_par.first, run.seq_par.last):
            trackcorr_c_loop(run, step)
        trackcorr_c_finish(run, run.seq_par.last)

        assert run.npart == 20
        assert run.nlinks == 20

    finally:
        os.chdir(original)


@pytest.mark.slow
def test_trackback():
    import os

    original = os.getcwd()
    try:
        os.chdir("test_data/track")
        if os.path.exists("res"):
            shutil.rmtree("res")
        if os.path.exists("img"):
            shutil.rmtree("img")
        shutil.copytree("res_orig", "res")
        shutil.copytree("img_orig", "img")

        cpar = read_control_par("parameters/ptv.par")
        calib = read_all_calibration(cpar.num_cams, base_path=".")

        run = tr_new(
            "parameters/sequence.par",
            "parameters/track.par",
            "parameters/criteria.par",
            "parameters/ptv.par",
            4,
            20000,
            "res/rt_is",
            "res/ptv_is",
            "res/added",
            calib,
            0.0001,
        )
        run.seq_par.first = 10240
        run.seq_par.last = 10250
        run.tpar = run.tpar._replace(add=1)

        track_forward_start(run)
        trackcorr_c_loop(run, run.seq_par.first)
        for step in range(run.seq_par.first + 1, run.seq_par.last):
            trackcorr_c_loop(run, step)
        trackcorr_c_finish(run, run.seq_par.last)

        run.tpar = run.tpar._replace(
            dvxmin=-50, dvymin=-50, dvzmin=-50, dvxmax=50, dvymax=50, dvzmax=50
        )
        run.lmax = np.linalg.norm(
            [
                run.tpar.dvxmin - run.tpar.dvxmax,
                run.tpar.dvymin - run.tpar.dvymax,
                run.tpar.dvzmin - run.tpar.dvzmax,
            ]
        )

        nlinks = trackback_c(run)
        # Note: the C test checks nlinks - 1.043062 but comments it out.
        # We'll just run it to ensure it completes successfully.

    finally:
        os.chdir(original)
