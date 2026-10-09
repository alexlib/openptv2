"""Backward tracking adds a particle only with a free blob in every camera.

The forward kernel (trackcorr_loop_fast) has required a free target in every
camera before adding a particle since the add-particle out-of-bounds fix;
trackback_loop_fast still followed liboptv's "at least two cameras".  The two
directions must use the same rule.

Scene: four cameras of test_data/test_cavity, one frame-1 particle with a link
to frame 0 and none to frame 2, an empty frame 2 whose only free blobs sit
exactly where the predicted position projects.
"""

from pathlib import Path

import numpy as np
import pytest

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.parameters import ControlPar
from openptv2.algorithms.track import _pack_cams_fast, _pack_cams_fast_tuples
from openptv2.algorithms.track_kernels_corr import trackback_loop_fast
from openptv2.algorithms.track_kernels_geom import point_to_pixel_fast

DATA = Path(__file__).resolve().parents[2] / "test_data" / "test_cavity"
POSI = 80
CAP = 8  # particle capacity of each frame


def _frame(num_cams):
    return dict(
        path_x=np.zeros((CAP, 3)),
        path_prev=np.full(CAP, -1, dtype=np.int32),
        path_next=np.full(CAP, -2, dtype=np.int32),
        path_inlist=np.zeros(CAP, dtype=np.int32),
        path_prio=np.full(CAP, 4, dtype=np.int32),
        path_finaldecis=np.zeros(CAP),
        path_decis=np.zeros((CAP, POSI)),
        path_linkdecis=np.full((CAP, POSI), -1, dtype=np.int32),
        corres_p=np.full((CAP, num_cams), -1, dtype=np.int32),
        corres_nr=np.zeros(CAP, dtype=np.int32),
        targ_x=np.zeros((num_cams, CAP)),
        targ_y=np.zeros((num_cams, CAP)),
        targ_tnr=np.full((num_cams, CAP), -1, dtype=np.int32),
        num_targets=np.zeros(num_cams, dtype=np.int32),
        num_parts=np.zeros(1, dtype=np.int32),
    )


def _run(cams_with_blob):
    cpar = ControlPar.from_yaml(str(DATA / "parameters.yaml"))
    nc = cpar.num_cams
    cals = [
        Calibration.from_file(
            str(DATA / "cal" / f"cam{i + 1}.tif.ori"),
            str(DATA / "cal" / f"cam{i + 1}.tif.addpar"),
        )
        for i in range(nc)
    ]
    cal_t, md_t, mo_t, mnr_t, mnz_t, mrw_t = _pack_cams_fast_tuples(
        *_pack_cams_fast(cals, cpar.mm)
    )
    cal_arr = np.asarray(list(cal_t), dtype=np.float64)
    md_arr = list(md_t)
    mo_arr = np.asarray(list(mo_t), dtype=np.float64)
    mnr_arr = np.array(list(mnr_t), dtype=np.int32)
    mnz_arr = np.array(list(mnz_t), dtype=np.int32)
    mrw_arr = np.array(list(mrw_t), dtype=np.float64)
    half = (cpar.imx / 2, cpar.imy / 2, 1 / cpar.pix_x, 1 / cpar.pix_y)

    f0, f1, f2, f3 = (_frame(nc) for _ in range(4))
    x1 = np.array([0.0, 0.0, 0.0])
    f1["path_x"][0] = x1
    f0["path_x"][0] = x1 + [1.0, 0.0, 0.0]
    f1["path_next"][0] = 0  # linked forward, nothing backward
    x2 = 2 * x1 - f0["path_x"][0]  # where trackback looks in frame 2

    # one free blob per chosen camera, exactly at x2's projection
    for ci in range(nc):
        px, py = point_to_pixel_fast(
            x2,
            cal_arr[ci],
            md_arr[ci],
            mo_arr[ci],
            mnr_arr[ci],
            mnz_arr[ci],
            mrw_arr[ci],
            int(mnr_arr[ci] > 0),
            *half,
            cpar.chfield,
        )
        assert 0 < px < cpar.imx and 0 < py < cpar.imy, f"x2 not seen by cam {ci}"
        if ci in cams_with_blob:
            f2["targ_x"][ci, 0], f2["targ_y"][ci, 0] = px, py
            f2["num_targets"][ci] = 1

    trackback_loop_fast(
        1,
        f0["path_x"],
        f1["path_x"],
        f1["path_prev"],
        f1["path_next"],
        f1["path_inlist"],
        f1["path_finaldecis"],
        f1["path_decis"],
        f1["path_linkdecis"],
        f2["path_x"],
        f2["path_prev"],
        f2["path_next"],
        f2["num_parts"],
        f2["targ_x"],
        f2["targ_y"],
        f2["targ_tnr"],
        f2["num_targets"],
        f2["corres_p"],
        f2["corres_nr"],
        f2["path_inlist"],
        f2["path_prio"],
        f2["path_finaldecis"],
        f2["path_decis"],
        f2["path_linkdecis"],
        f3["path_x"],
        f3["path_prev"],
        cal_arr,
        md_arr,
        mo_arr,
        mnr_arr,
        mnz_arr,
        mrw_arr,
        -5.0,
        5.0,
        -5.0,
        5.0,
        -5.0,
        5.0,  # dv bounds
        1.0,
        30.0,  # dacc, dangle
        1,  # add
        np.sqrt(3 * 10.0**2),  # lmax
        -100.0,
        100.0,
        -100.0,
        100.0,
        -100.0,
        100.0,  # X_lay, Y, Z_lay bounds
        nc,
        *half,
        cpar.chfield,
        float(cpar.imx),
        float(cpar.imy),
        cpar.pix_x,
        cpar.pix_y,
        1e-5,
    )
    # num_added also counts links made in the link-resolution step (as in
    # liboptv), so new particles are read from frame 2 itself.
    return f2


@pytest.mark.parametrize("cams", [(0, 1), (0, 1, 2)])
def test_trackback_does_not_add_with_a_camera_missing(cams):
    f2 = _run(cams)
    assert f2["num_parts"][0] == 0


def test_trackback_adds_with_a_blob_in_every_camera():
    f2 = _run((0, 1, 2, 3))
    assert f2["num_parts"][0] == 1
    assert (f2["corres_p"][0] >= 0).all()


def test_trackback_c_passes_the_volume_y_limits(monkeypatch, tmp_path):
    """trackback_c must hand the kernel the run's Y limits (from
    volumedimension), as the forward loop does. liboptv's trackback_c left
    them at 0 -- 3DPTV filled them with volumedimension() -- so the volume
    check 0 < y < 0 never passed and backward tracking never added."""
    import shutil
    import sys

    import openptv2.algorithms.track as track
    from openptv2.algorithms.parameters import SequencePar, TrackPar, VolumePar
    from openptv2.algorithms.tracking_run import tr_new

    sys.path.insert(0, str(Path(__file__).parent))
    from test_track import read_all_calibration

    src = Path(__file__).resolve().parents[2] / "test_data" / "track"
    if not (src / "res_orig").exists() or not (src / "img_orig").exists():
        pytest.skip("test_data/track fixtures missing")
    work = tmp_path / "track"
    shutil.copytree(src, work)
    shutil.copytree(work / "res_orig", work / "res", dirs_exist_ok=True)
    shutil.copytree(work / "img_orig", work / "img", dirs_exist_ok=True)
    monkeypatch.chdir(work)

    seen = []

    def spy(*args):
        seen.append((args[44], args[45]))  # ymin, ymax
        return 0, 0

    monkeypatch.setattr(track, "_trackback_loop_fast", spy)
    cpar = ControlPar.from_yaml("parameters.yaml")
    run = tr_new(
        SequencePar.from_yaml("parameters.yaml"),
        TrackPar.from_yaml("parameters.yaml"),
        VolumePar.from_yaml("parameters.yaml"),
        cpar,
        4,
        20000,
        "res/rt_is",
        "res/ptv_is",
        "res/added",
        read_all_calibration(cpar.num_cams, base_path="."),
        0.0001,
    )
    run.seq_par.first, run.seq_par.last = 10240, 10250
    track.trackback_c(run)

    assert seen, "trackback kernel was not called"
    assert run.ymin < run.ymax
    assert all(lim == (run.ymin, run.ymax) for lim in seen)
