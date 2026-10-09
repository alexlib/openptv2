"""The tracking kernels' multimedia LUT lookup must agree with img_coord.

imgcoord._get_mmf_from_mmlut_core and multimed.get_mmf_from_mmlut reject a
point unless all four bilinear corners (ir..ir+1, iz..iz+1) are inside the
table, and fall back to the iterative solver otherwise.  The three inlined
copies in the tracking kernels kept liboptv's older test (ir <= nr,
iz <= nz, last index <= nr*nz), which accepts the table's top row and the row
above it: the "z+1" corners then wrap into the next radial row (and at one
corner read one element past the end).  Points in that band were projected
16-35 px off.
"""

from pathlib import Path

import numpy as np
import pytest

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.imgcoord import img_coord
from openptv2.algorithms.multimed import init_mmlut
from openptv2.algorithms.parameters import ControlPar, MmNp, VolumePar
from openptv2.algorithms.trafo import metric_to_pixel

CAL_DIR = Path(__file__).resolve().parents[2] / "test_data" / "track" / "cal"
PIX = 0.012


@pytest.fixture(scope="module")
def water_rig():
    cal = Calibration()
    cal.from_file(str(CAL_DIR / "cam1.tif.ori"), str(CAL_DIR / "cam1.tif.addpar"))
    mm = MmNp(n1=1.0, n2=[1.49], d=[5.0], n3=1.33)
    cpar = ControlPar(
        num_cams=1,
        imx=1280,
        imy=1024,
        pix_x=PIX,
        pix_y=PIX,
        mm=mm,
        chfield=0,
        tiff_flag=1,
        hp_flag=1,
        allCam_flag=0,
        img_base_name=[""],
        cal_img_base_name=[""],
    )
    vpar = VolumePar(X_lay=[-100, 100], Zmin_lay=[-100, -100], Zmax_lay=[30, 30])
    init_mmlut(vpar, cpar, cal)
    assert cal.mmlut.data is not None and len(cal.mmlut.data) > 0
    # Points throughout the volume and past both Z ends of the table; the
    # table's top rows are where the old bounds test went wrong.
    rng = np.random.default_rng(0)
    pts = np.column_stack(
        [
            rng.uniform(-150, 150, 1500),
            rng.uniform(-150, 150, 1500),
            rng.uniform(-120, 60, 1500),
        ]
    )
    exact = np.array([metric_to_pixel(*img_coord(p, cal, mm), cpar) for p in pts])
    return cal, mm, cpar, pts, exact


def _max_error(project, pts, exact):
    got = np.array([project(p) for p in pts])
    return np.hypot(*(got - exact).T).max()


def test_track_point_to_pixel_matches_img_coord(water_rig):
    from openptv2.algorithms.track import _point_to_pixel_fast

    cal, mm, cpar, pts, exact = water_rig
    err = _max_error(
        lambda p: _point_to_pixel_fast(p, cal, cpar.imx, cpar.imy, PIX, PIX, 0, mm),
        pts,
        exact,
    )
    assert err < 1e-3, f"track._point_to_pixel_fast off by {err:.3f} px"


def test_kernel_geom_point_to_pixel_matches_img_coord(water_rig):
    from openptv2.algorithms.track_kernels import pack_cal_array, pack_mmlut
    from openptv2.algorithms.track_kernels_geom import point_to_pixel_fast

    cal, mm, cpar, pts, exact = water_rig
    pc, lut = pack_cal_array(cal, mm), pack_mmlut(cal)
    err = _max_error(
        lambda p: point_to_pixel_fast(p, pc, *lut, 1, 640, 512, 1 / PIX, 1 / PIX, 0),
        pts,
        exact,
    )
    assert err < 1e-3, f"track_kernels_geom.point_to_pixel_fast off by {err:.3f} px"


def test_kernel_pixel_point_to_pixel_out_matches_img_coord(water_rig):
    from openptv2.algorithms.track_kernels import pack_cal_array, pack_mmlut
    from openptv2.algorithms.track_kernels_pixel import _point_to_pixel_out

    cal, mm, cpar, pts, exact = water_rig
    pc, lut = pack_cal_array(cal, mm), pack_mmlut(cal)
    out = np.zeros(2)

    def project(p):
        _point_to_pixel_out(p, pc, *lut, 1, 640, 512, 1 / PIX, 1 / PIX, 0, out)
        return out.copy()

    err = _max_error(project, pts, exact)
    assert err < 1e-3, f"track_kernels_pixel._point_to_pixel_out off by {err:.3f} px"


def test_kernels_without_lut_match_img_coord():
    """No LUT built: the kernels must still apply refraction.

    pack_mmlut used to hand the kernels a 2x2 table of ones and rely on an
    exact 1.0 result to trigger the iterative solve; bilinear interpolation
    of ones is not always exactly 1.0, and then refraction was skipped (up to
    186 px off on the cavity cameras).
    """
    from openptv2.algorithms.track import _pack_cams_fast, _pack_cams_fast_tuples
    from openptv2.algorithms.track_kernels_geom import point_to_pixel_fast

    data = Path(__file__).resolve().parents[2] / "test_data" / "test_cavity"
    cpar = ControlPar.from_yaml(str(data / "parameters.yaml"))
    cals = [
        Calibration.from_file(
            str(data / "cal" / f"cam{i + 1}.tif.ori"),
            str(data / "cal" / f"cam{i + 1}.tif.addpar"),
        )
        for i in range(cpar.num_cams)
    ]
    assert all(c.mmlut.data is None or len(c.mmlut.data) == 0 for c in cals)
    cal_t, md_t, mo_t, mnr_t, mnz_t, mrw_t = _pack_cams_fast_tuples(
        *_pack_cams_fast(cals, cpar.mm)
    )
    half = (cpar.imx / 2, cpar.imy / 2, 1 / cpar.pix_x, 1 / cpar.pix_y)
    rng = np.random.default_rng(1)
    worst = 0.0
    for p in rng.uniform(-30, 30, (100, 3)):
        for ci, cal in enumerate(cals):
            exact = np.array(metric_to_pixel(*img_coord(p, cal, cpar.mm), cpar))
            got = np.array(
                point_to_pixel_fast(
                    p,
                    np.asarray(cal_t[ci]),
                    md_t[ci],
                    np.asarray(mo_t[ci]),
                    mnr_t[ci],
                    mnz_t[ci],
                    mrw_t[ci],
                    int(mnr_t[ci] > 0),
                    *half,
                    0,
                )
            )
            worst = max(worst, np.hypot(*(got - exact)))
    assert worst < 1e-3, f"kernel without LUT off by {worst:.2f} px"


def _lut_vs_iterative(r_start_cells):
    """Worst |LUT - iterative| radial factor over the table, from radius
    r_start_cells * rw outwards, for a test_cavity camera in water."""
    from openptv2.algorithms.multimed import (
        get_mmf_from_mmlut,
        multimed_r_nlay_iterative,
        trans_cam_point,
    )
    from openptv2.algorithms.parameters import VolumePar

    data_dir = Path(__file__).resolve().parents[2] / "test_data" / "test_cavity"
    cpar = ControlPar.from_yaml(str(data_dir / "parameters.yaml"))
    vpar = VolumePar.from_yaml(str(data_dir / "parameters.yaml"))
    cal = Calibration.from_file(
        str(data_dir / "cal" / "cam1.tif.ori"),
        str(data_dir / "cal" / "cam1.tif.addpar"),
    )
    mm = cpar.mm
    init_mmlut(vpar, cpar, cal)
    lut = cal.mmlut
    # the glass frame the table is built in: camera on the axis at z0_t
    *_, z0_t = trans_cam_point(
        np.zeros(3),
        cal.ext_par.x0,
        cal.ext_par.y0,
        cal.ext_par.z0,
        cal.glass_par.vec_x,
        cal.glass_par.vec_y,
        cal.glass_par.vec_z,
        mm.n1,
        mm.n2[0],
        mm.n3,
        mm.d[0],
    )
    worst = 0.0
    # r = 0 itself is skipped: there the solver returns a placeholder 1.0,
    # and any factor is exact because it multiplies r = 0.
    r0 = max(r_start_cells * lut.rw, lut.rw / 6)
    for R in np.arange(r0, (lut.nr - 1) * lut.rw, lut.rw / 3):
        for Z in np.arange(0.0, (lut.nz - 1) * lut.rw, lut.rw / 3):
            pos = np.array([lut.origin[0] + R, lut.origin[1], lut.origin[2] + Z])
            got = get_mmf_from_mmlut(pos, lut.origin, lut.nr, lut.nz, lut.rw, lut.data)
            if got <= 0:
                continue
            # converged reference: the default stop (|rdiff| < 0.001 mm) is a
            # relative error of ~1e-4 close to the axis
            exact = multimed_r_nlay_iterative(
                pos[0],
                pos[1],
                pos[2],
                0.0,
                0.0,
                z0_t,
                mm.n1,
                mm.n2[0],
                mm.n3,
                mm.d[0],
                1,
                None,
                None,
                200,
                1e-9,
            )
            worst = max(worst, abs(got - exact))
    return worst


def test_lut_matches_iterative_away_from_axis():
    assert _lut_vs_iterative(1) < 1e-4


def test_lut_matches_iterative_near_axis():
    """The R = 0 row used to hold the solver's r == 0 placeholder 1.0 instead
    of the r -> 0 limit, so the first radial cell interpolated toward 1.0."""
    assert _lut_vs_iterative(0) < 1e-4
