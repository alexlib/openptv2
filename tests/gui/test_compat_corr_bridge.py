"""
Precise test: does compat correspondences pass correct frame/corrected to raw corr?

Usage:
    uv run pytest tests/gui/test_compat_corr_bridge.py -v -s
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest


def _prepare_test_data(test_dir):
    img_dir = test_dir / "img"
    img_orig = test_dir / "img_orig"
    if img_dir.exists():
        shutil.rmtree(img_dir)
    shutil.copytree(img_orig, img_dir)


@pytest.fixture(scope="module")
def cavity_dir():
    from tests._support import find_test_data_root

    root = find_test_data_root(Path(__file__))
    d = root / "test_cavity"
    if not d.exists():
        pytest.skip(f"Test data dir {d} not found")
    return d


def test_compat_bridge_correctness(cavity_dir):
    """Test that compat correspondences passes correct data to raw corr."""
    try:
        import optv
    except ImportError:
        pytest.skip("optv bindings not available")

    _prepare_test_data(cavity_dir)
    original_cwd = Path.cwd()
    os.chdir(cavity_dir)

    try:
        from imageio.v3 import imread
        from skimage.color import rgb2gray
        from skimage.util import img_as_ubyte

        from openptv2.gui.experiment import Experiment
        from openptv2.gui.ptv import py_start_proc_c, simple_highpass

        yaml_file = cavity_dir / "parameters_Run1.yaml"
        exp = Experiment()
        exp.pm.from_yaml(yaml_file)
        num_cams = exp.pm.num_cams
        cpar, spar, vpar, trk, tpar, cals_optv, epar = py_start_proc_c(exp.pm)

        # Build compat params
        ptv_p = exp.pm.get_parameter("ptv")
        crit_p = exp.pm.get_parameter("criteria")
        targ_p = exp.pm.get_parameter("targ_rec")
        cal_p = exp.pm.get_parameter("cal_ori")

        from openptv2.calibration import Calibration as CalC
        from openptv2.parameters import (
            ControlParams as C,
        )
        from openptv2.parameters import (
            TargetParams as T,
        )
        from openptv2.parameters import (
            VolumeParams as V,
        )

        cp_c = C(num_cams=num_cams)
        cp_c.set_image_size((ptv_p["imx"], ptv_p["imy"]))
        cp_c.set_pixel_size((ptv_p["pix_x"], ptv_p["pix_y"]))
        cp_c.set_hp_flag(ptv_p["hp_flag"])
        cp_c.set_allCam_flag(ptv_p["allcam_flag"])
        cp_c.set_tiff_flag(ptv_p["tiff_flag"])
        cp_c.set_chfield(ptv_p["chfield"])
        mm = cp_c.get_multimedia_params()
        mm.set_n1(ptv_p["mmp_n1"])
        mm.set_layers([ptv_p["mmp_n2"]], [ptv_p["mmp_d"]])
        mm.set_n3(ptv_p["mmp_n3"])

        vp_c = V()
        vp_c.set_X_lay(crit_p["X_lay"])
        vp_c.set_Zmin_lay(crit_p["Zmin_lay"])
        vp_c.set_Zmax_lay(crit_p["Zmax_lay"])
        vp_c.set_cn(crit_p["cn"])
        vp_c.set_cnx(crit_p.get("cnx", 0))
        vp_c.set_cny(crit_p.get("cny", 0))
        vp_c.set_csumg(crit_p.get("csumg", 0))
        vp_c.set_eps0(crit_p.get("eps0", 0))
        vp_c.set_corrmin(crit_p["corrmin"])

        tp_c = T()
        tp_c.set_grey_thresholds(targ_p["gvthres"])
        tp_c.set_max_discontinuity(targ_p["disco"])
        tp_c.set_pixel_count_bounds((targ_p["nnmin"], targ_p["nnmax"]))
        tp_c.set_xsize_bounds((targ_p["nxmin"], targ_p["nxmax"]))
        tp_c.set_ysize_bounds((targ_p["nymin"], targ_p["nymax"]))
        tp_c.set_min_sum_grey(targ_p["sumg_min"])
        tp_c.set_cross_size(targ_p["cr_sz"])

        cals_c = []
        for i in range(num_cams):
            cal = CalC()
            cal.from_file(
                cal_p["img_ori"][i], cal_p["img_ori"][i].replace(".ori", ".addpar")
            )
            cals_c.append(cal)

        frame = 10000

        # Get compat detections
        from openptv2.algorithms.tracking_frame_buf import Frame as RawFrame
        from openptv2.correspondences import MatchedCoords as c_mc
        from openptv2.segmentation import target_recognition as c_tr

        det_c, corr_c = [], []
        for i_cam in range(num_cams):
            imname = Path(spar.get_img_base_name(i_cam) % frame)
            img = imread(imname)
            if img.ndim > 2:
                img = rgb2gray(img)
            if img.dtype != np.uint8:
                img = img_as_ubyte(img)
            high_pass = simple_highpass(img, cpar)
            targs = c_tr(high_pass, tp_c, i_cam, cp_c)
            targs.sort_y()
            det_c.append(targs)
            corr_c.append(c_mc(targs, cp_c, cals_c[i_cam]))

        # --- Test 1: compat corr internal call ---
        from openptv2.algorithms.correspondences import correspondences as raw_corr

        # Build frame from compat detections (same as compat corr does internally)
        frame_a = RawFrame(num_cams=num_cams, max_targets=1000)
        for cam in range(num_cams):
            if hasattr(det_c[cam], "_targets"):
                targets = det_c[cam]._targets
            else:
                targets = det_c[cam]
            frame_a.targets[cam] = targets
            frame_a.num_targets[cam] = len(targets)

        corrected_a = [mc._corrected for mc in corr_c]
        unwrapped_cals = [c._cal if hasattr(c, "_cal") else c for c in cals_c]

        raw_vpar = vp_c._vpar if hasattr(vp_c, "_vpar") else vp_c
        raw_cpar = cp_c._cpar if hasattr(cp_c, "_cpar") else cp_c

        ntupels_a, mc_a = raw_corr(
            frame_a, corrected_a, raw_vpar, raw_cpar, unwrapped_cals
        )
        print(
            f"\n  INTERNAL raw_corr(frame from compat detections): "
            f"total={mc_a[3]} match_counts={mc_a}"
        )

        # --- Test 2: raw corr with raw params but compat detections ---
        from openptv2.algorithms.calibration import Calibration
        from openptv2.algorithms.parameters import ControlPar, VolumePar

        cpar_r = ControlPar(num_cams=num_cams)
        cpar_r.imx = ptv_p["imx"]
        cpar_r.imy = ptv_p["imy"]
        cpar_r.pix_x = ptv_p["pix_x"]
        cpar_r.pix_y = ptv_p["pix_y"]
        cpar_r.hp_flag = ptv_p["hp_flag"]
        cpar_r.allCam_flag = ptv_p["allcam_flag"]
        cpar_r.tiff_flag = ptv_p["tiff_flag"]
        cpar_r.chfield = ptv_p["chfield"]
        cpar_r.mm.n1 = ptv_p["mmp_n1"]
        cpar_r.mm.n3 = ptv_p["mmp_n3"]
        cpar_r.mm.n2[:] = [ptv_p["mmp_n2"], 0.0, 0.0]
        cpar_r.mm.d[:] = [ptv_p["mmp_d"], 0.0, 0.0]

        vpar_r = VolumePar()
        vpar_r.X_lay = np.array(crit_p["X_lay"])
        vpar_r.Zmin_lay = np.array(crit_p["Zmin_lay"])
        vpar_r.Zmax_lay = np.array(crit_p["Zmax_lay"])
        vpar_r.cn = crit_p["cn"]
        vpar_r.cnx = crit_p.get("cnx", 0)
        vpar_r.cny = crit_p.get("cny", 0)
        vpar_r.csumg = crit_p.get("csumg", 0)
        vpar_r.eps0 = crit_p.get("eps0", 0)
        vpar_r.corrmin = crit_p["corrmin"]

        cals_r = [
            Calibration.from_file(
                cal_p["img_ori"][i], cal_p["img_ori"][i].replace(".ori", ".addpar")
            )
            for i in range(num_cams)
        ]

        ntupels_b, mc_b = raw_corr(frame_a, corrected_a, vpar_r, cpar_r, cals_r)
        print(
            f"  INTERNAL raw_corr(raw params, compat detections): "
            f"total={mc_b[3]} match_counts={mc_b}"
        )

        # --- COMPARE: frame/corrected from raw vs compat ---
        print("\n  Frame comparison (compat build vs optv target based):")
        for cam in range(num_cams):
            print(
                f"    cam{cam + 1}: num_targets={frame_a.num_targets[cam]} "
                f"corrected_len={len(corrected_a[cam])}"
            )

        if mc_b[3] != mc_a[3]:
            print(
                f"\n  >>> MISMATCH: raw params ({mc_b[3]}) vs compat params ({mc_a[3]}) <<<"
            )
            # The parameter objects differ — check key fields
            raw_cpar = cp_c._cpar if hasattr(cp_c, "_cpar") else cp_c
            raw_vpar = vp_c._vpar if hasattr(vp_c, "_vpar") else vp_c
            print(f"  cpar_c.imx={raw_cpar.imx} cpar_r.imx={cpar_r.imx}")
            print(f"  cpar_c.imx==cpar_r.imx: {raw_cpar.imx == cpar_r.imx}")
            print(f"  vp_c.corrmin={raw_vpar.corrmin} vpar_r.corrmin={vpar_r.corrmin}")
            print(f"  mm nlay: compat={raw_cpar.mm.nlay} raw={cpar_r.mm.nlay}")
            print(f"  mm n1: compat={raw_cpar.mm.n1} raw={cpar_r.mm.n1}")
            print(f"  mm n2: compat={raw_cpar.mm.n2} raw={cpar_r.mm.n2}")
            print(f"  mm d: compat={raw_cpar.mm.d} raw={cpar_r.mm.d}")
            print(f"  mm n3: compat={raw_cpar.mm.n3} raw={cpar_r.mm.n3}")

    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
