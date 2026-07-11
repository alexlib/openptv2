"""Multimedia (refraction) parity + ground-truth, openptv2.algorithms vs optv.

The existing imgcoord/point_positions parity tests all run with refraction OFF
(n1=n2=n3=1), so the multimed / through-media ray_tracing path was never checked
against liboptv with realistic optics. This test drives the FULL projection and
triangulation with the real cavity multimedia model (water/glass), feeding the
SAME parameters — read from the dataset YAML — into both libraries:

  1. forward: project the known 3D calibration body to pixels (image_coordinates)
     -> our img_coord must match optv to ~1e-6.
  2. inverse: triangulate those projections back (point_positions)
     -> our result must match optv AND recover the original 3D (ground truth).

Because there is no standalone optv `multimed`/`ray_tracing` module, this is the
way to parity-check refraction: it is exercised transitively here.
"""
from pathlib import Path

import numpy as np
import pytest
import yaml

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.imgcoord import img_coord
from openptv2.algorithms.orientation import point_position
from openptv2.algorithms.parameters import MultimediaPar

DATASET = Path("test_data/test_cavity")


def _has_optv() -> bool:
    try:
        import optv.imgcoord  # noqa: F401
        import optv.orientation  # noqa: F401
        return True
    except ImportError:
        return False


def _load_yaml_ptv():
    y = yaml.safe_load((DATASET / "parameters_Run1.yaml").read_text())
    return int(y.get("num_cams") or y["ptv"]["num_cams"]), y["ptv"], y["criteria"]


def _body_points() -> np.ndarray:
    data = np.loadtxt(DATASET / "cal" / "target_on_a_side.txt", ndmin=2)
    return np.ascontiguousarray(data[:, 1:4], dtype=float)


@pytest.mark.parity
@pytest.mark.skipif(not _has_optv(), reason="optv (Cython bindings) not available")
def test_multimedia_projection_and_triangulation_parity():
    from optv.calibration import Calibration as CCalib
    from optv.imgcoord import image_coordinates
    from optv.orientation import point_positions
    from optv.parameters import ControlParams, VolumeParams

    num_cams, ptv, crit = _load_yaml_ptv()
    n1, n2, n3, d = (
        float(ptv["mmp_n1"]), float(ptv["mmp_n2"]),
        float(ptv["mmp_n3"]), float(ptv["mmp_d"]),
    )
    assert n2 != 1.0 and n3 != 1.0, "expected real refraction in this dataset"

    # --- build identical params for both libs, from the YAML ---
    c_cpar = ControlParams(num_cams)
    c_cpar.set_image_size((int(ptv["imx"]), int(ptv["imy"])))
    c_cpar.set_pixel_size((float(ptv["pix_x"]), float(ptv["pix_y"])))
    c_mm = c_cpar.get_multimedia_params()
    c_mm.set_n1(n1)
    c_mm.set_n3(n3)
    c_mm.set_layers(np.array([n2]), np.array([d]))

    py_mm = MultimediaPar(nlay=1, n1=n1, n2=[n2], d=[d], n3=n3)

    c_vpar = VolumeParams()
    c_vpar.set_X_lay(np.array(crit["X_lay"], dtype=float))
    c_vpar.set_Zmin_lay(np.array(crit["Zmin_lay"], dtype=float))
    c_vpar.set_Zmax_lay(np.array(crit["Zmax_lay"], dtype=float))
    c_vpar.set_cn(float(crit["cn"]))
    c_vpar.set_cnx(float(crit["cnx"]))
    c_vpar.set_cny(float(crit["cny"]))
    c_vpar.set_csumg(float(crit["csumg"]))
    c_vpar.set_corrmin(float(crit["corrmin"]))
    c_vpar.set_eps0(float(crit["eps0"]))

    c_cals, py_cals = [], []
    for cam in range(num_cams):
        ori = str(DATASET / "cal" / f"cam{cam + 1}.tif.ori")
        add = str(DATASET / "cal" / f"cam{cam + 1}.tif.addpar")
        cc = CCalib()
        cc.from_file(ori_file=ori, add_file=add)
        c_cals.append(cc)
        py_cals.append(Calibration.from_file(ori, add))

    xyz = _body_points()
    npts = len(xyz)

    # --- 1. forward projection parity (metric), refraction ON ---
    per_cam_metric = []
    for cam in range(num_cams):
        c_xy = image_coordinates(xyz, c_cals[cam], c_mm)          # (npts, 2)
        py_xy = np.array([img_coord(xyz[i], py_cals[cam], py_mm)
                          for i in range(npts)])
        np.testing.assert_allclose(
            py_xy, c_xy, atol=1e-6,
            err_msg=f"cam{cam + 1}: img_coord (multimedia) differs from optv",
        )
        per_cam_metric.append(c_xy)

    # --- 2. triangulation parity + ground-truth recovery ---
    c_targs = np.array(per_cam_metric).transpose(1, 0, 2)          # (npts, ncam, 2)
    c_res, _c_rcm = point_positions(c_targs, c_cpar, c_cals, c_vpar)

    py_res = np.array([
        point_position(c_targs[i], num_cams, py_mm, py_cals)[0]
        for i in range(npts)
    ])

    np.testing.assert_allclose(
        py_res, c_res, atol=1e-6,
        err_msg="point_positions (multimedia) differs from optv",
    )
    # Ground truth: project-then-triangulate must recover the body coords.
    # Tolerance reflects the real calibration's residual (~1.8 px RMS ->
    # ~0.05 mm object space) plus the iterative distortion solver's finite
    # tolerance; a perfect (synthetic) cal recovers to ~1e-6, but this fixture
    # carries a realistic calibration. The parity checks above (1e-6 vs optv)
    # are the strict correctness guard; this only rejects gross errors.
    np.testing.assert_allclose(
        py_res, xyz, atol=0.1,
        err_msg="round-trip did not recover known 3D body points",
    )
