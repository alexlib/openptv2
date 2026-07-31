"""
Pinpoint where the compat correspondences wrapper loses matches.

Usage:
    uv run pytest tests/gui/test_compat_correspondence_bug.py -v -s
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


def test_compat_correspondences_parity(cavity_dir):
    """Compare compat correspondences vs optv, debugging at each step."""
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
        cpar, spar, vpar, trk_par, tpar, cals_optv, epar = py_start_proc_c(exp.pm)

        # Build compat params
        from openptv2.calibration import Calibration as CCal
        from openptv2.parameters import (
            ControlParams as C,
        )
        from openptv2.parameters import (
            TargetParams as T,
        )
        from openptv2.parameters import (
            VolumeParams as V,
        )

        ptv_p = exp.pm.get_parameter("ptv")
        seq_p = exp.pm.get_parameter("sequence")
        crit_p = exp.pm.get_parameter("criteria")
        targ_p = exp.pm.get_parameter("targ_rec")
        cal_p = exp.pm.get_parameter("cal_ori")

        cpar_c = C(num_cams=num_cams)
        cpar_c.set_image_size((ptv_p["imx"], ptv_p["imy"]))
        cpar_c.set_pixel_size((ptv_p["pix_x"], ptv_p["pix_y"]))
        cpar_c.set_hp_flag(ptv_p["hp_flag"])
        cpar_c.set_allCam_flag(ptv_p["allcam_flag"])
        cpar_c.set_tiff_flag(ptv_p["tiff_flag"])
        cpar_c.set_chfield(ptv_p["chfield"])
        mm = cpar_c.get_multimedia_params()
        mm.set_n1(ptv_p["mmp_n1"])
        mm.set_layers([ptv_p["mmp_n2"]], [ptv_p["mmp_d"]])
        mm.set_n3(ptv_p["mmp_n3"])

        vpar_c = V()
        vpar_c.set_X_lay(crit_p["X_lay"])
        vpar_c.set_Zmin_lay(crit_p["Zmin_lay"])
        vpar_c.set_Zmax_lay(crit_p["Zmax_lay"])
        vpar_c.set_cn(crit_p["cn"])
        vpar_c.set_cnx(crit_p.get("cnx", 0))
        vpar_c.set_cny(crit_p.get("cny", 0))
        vpar_c.set_csumg(crit_p.get("csumg", 0))
        vpar_c.set_eps0(crit_p.get("eps0", 0))
        vpar_c.set_corrmin(crit_p["corrmin"])

        tpar_c = T()
        tpar_c.set_grey_thresholds(targ_p["gvthres"])
        tpar_c.set_max_discontinuity(targ_p["disco"])
        tpar_c.set_pixel_count_bounds((targ_p["nnmin"], targ_p["nnmax"]))
        tpar_c.set_xsize_bounds((targ_p["nxmin"], targ_p["nxmax"]))
        tpar_c.set_ysize_bounds((targ_p["nymin"], targ_p["nymax"]))
        tpar_c.set_min_sum_grey(targ_p["sumg_min"])
        tpar_c.set_cross_size(targ_p["cr_sz"])

        cals_c = []
        for i in range(num_cams):
            cal = CCal()
            ori_f = cal_p["img_ori"][i]
            add_f = ori_f.replace(".ori", ".addpar")
            cal.from_file(ori_f, add_f)
            cals_c.append(cal)

        frame = 10000

        print(f"\n{'=' * 60}")
        print(f"Frame {frame}: step-by-step compat correspondences debug")
        print(f"{'=' * 60}")

        # Get optv detections (same raw input for both)
        det_o, corr_o = [], []
        for i_cam in range(num_cams):
            imname = Path(spar.get_img_base_name(i_cam) % frame)
            img = imread(imname)
            if img.ndim > 2:
                img = rgb2gray(img)
            if img.dtype != np.uint8:
                img = img_as_ubyte(img)
            high_pass = simple_highpass(img, cpar)
            from optv.segmentation import target_recognition as o_tr

            targs = o_tr(high_pass, tpar, i_cam, cpar)
            targs.sort_y()
            det_o.append(targs)
            from optv.correspondences import MatchedCoords as o_mc

            corr_o.append(o_mc(targs, cpar, cals_optv[i_cam]))

        # OPTV correspondences
        from optv.correspondences import correspondences as o_corr

        sp_o, sc_o, nt_o = o_corr(det_o, corr_o, cals_optv, vpar, cpar)
        total_o = sum(x.shape[1] for x in sp_o if x is not None)
        print(f"  OPTV: total={total_o}")

        # Build raw algorithms Frame + corrected from optv detections
        from openptv2.algorithms.epi import Coord2d
        from openptv2.algorithms.tracking_frame_buf import Frame as RawFrame

        raw_frm = RawFrame(num_cams=num_cams, max_targets=10000)
        raw_corrected = []
        for i_cam in range(num_cams):
            n = len(det_o[i_cam])
            raw_frm.num_targets[i_cam] = n
            for tnum in range(n):
                t = det_o[i_cam][tnum]
                raw_frm.targets[i_cam][tnum].pnr = t.pnr()
                raw_frm.targets[i_cam][tnum].x = t.pos()[0]
                raw_frm.targets[i_cam][tnum].y = t.pos()[1]
                raw_frm.targets[i_cam][tnum].n = t.count_pixels()[0]
                raw_frm.targets[i_cam][tnum].nx = t.count_pixels()[1]
                raw_frm.targets[i_cam][tnum].ny = t.count_pixels()[2]
                raw_frm.targets[i_cam][tnum].sumg = t.sum_grey_value()
                raw_frm.targets[i_cam][tnum].tnr = t.tnr()

            pos_optv, pnr_optv = corr_o[i_cam].as_arrays()
            raw_corrected.append(
                [
                    Coord2d(x=pos_optv[j, 0], y=pos_optv[j, 1], pnr=pnr_optv[j])
                    for j in range(len(pos_optv))
                ]
            )

        # Call raw algorithms correspondences
        from openptv2.algorithms.calibration import Calibration
        from openptv2.algorithms.correspondences import correspondences as raw_corr
        from openptv2.algorithms.parameters import ControlPar, VolumePar

        # Raw params from same data
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

        cals_r = []
        for i in range(num_cams):
            ori_f = cal_p["img_ori"][i]
            add_f = ori_f.replace(".ori", ".addpar")
            cals_r.append(Calibration.from_file(ori_f, add_f))

        ntupels_raw, mc_raw = raw_corr(raw_frm, raw_corrected, vpar_r, cpar_r, cals_r)
        print(f"  RAW algs: total={mc_raw[3]} match_counts={mc_raw}")

        # --- COMPAT correspondences (calls raw under the hood) ---
        print("\n  --- COMPAT LAYER ---")
        from openptv2.correspondences import MatchedCoords as comp_mc
        from openptv2.correspondences import correspondences as comp_corr
        from openptv2.segmentation import target_recognition as comp_tr

        # Build detection + corrected using COMPAT APIs
        det_c, corr_c = [], []
        for i_cam in range(num_cams):
            imname = Path(spar.get_img_base_name(i_cam) % frame)
            img = imread(imname)
            if img.ndim > 2:
                img = rgb2gray(img)
            if img.dtype != np.uint8:
                img = img_as_ubyte(img)
            high_pass = simple_highpass(img, cpar)

            targs_c = comp_tr(high_pass, tpar_c, i_cam, cpar_c)
            targs_c.sort_y()
            det_c.append(targs_c)
            mc_c = comp_mc(targs_c, cpar_c, cals_c[i_cam])
            corr_c.append(mc_c)
            print(f"    cam{i_cam + 1}: {len(targs_c)} targets (compat)")

        sp_c, sc_c, nt_c = comp_corr(det_c, corr_c, cals_c, vpar_c, cpar_c)
        total_c = sum(x.shape[1] for x in sp_c if x is not None)
        print(f"  COMPAT: total={total_c}")

        # Also test: compat correspondences but using OPTV detections
        print("\n  --- COMPAT layer using OPTV detections ---")
        det_w, corr_w = [], []
        for i_cam in range(num_cams):
            targs_o = det_o[i_cam]  # optv TargetArray
            det_w.append(targs_o)
            mc_w = comp_mc(targs_o, cpar_c, cals_c[i_cam])
            corr_w.append(mc_w)
            print(f"    cam{i_cam + 1}: {len(targs_o)} targets (opTV -> compat)")

        sp_w, sc_w, nt_w = comp_corr(det_w, corr_w, cals_c, vpar_c, cpar_c)
        total_w = sum(x.shape[1] for x in sp_w if x is not None)
        print(f"  COMPAT+optv targets: total={total_w}")

        # Test: raw correspondences using COMPAT params (unwrap ._cal, ._cpar, etc)
        print("\n  --- RAW algs using COMPAT-unwrapped params ---")
        from openptv2.algorithms.epi import Coord2d
        from openptv2.algorithms.tracking_frame_buf import Frame as RFrame

        r_frm = RFrame(num_cams=num_cams, max_targets=10000)
        r_corr = []
        for i_cam in range(num_cams):
            targs_cam = det_o[i_cam]  # optv targets
            n = len(targs_cam)
            r_frm.num_targets[i_cam] = n
            for tnum in range(n):
                t = targs_cam[tnum]
                r_frm.targets[i_cam][tnum].pnr = t.pnr()
                r_frm.targets[i_cam][tnum].x = t.pos()[0]
                r_frm.targets[i_cam][tnum].y = t.pos()[1]
                r_frm.targets[i_cam][tnum].n = t.count_pixels()[0]
                r_frm.targets[i_cam][tnum].nx = t.count_pixels()[1]
                r_frm.targets[i_cam][tnum].ny = t.count_pixels()[2]
                r_frm.targets[i_cam][tnum].sumg = t.sum_grey_value()
                r_frm.targets[i_cam][tnum].tnr = t.tnr()
            pos_o, pnr_o = corr_o[i_cam].as_arrays()
            r_corr.append(
                [
                    Coord2d(x=pos_o[j, 0], y=pos_o[j, 1], pnr=pnr_o[j])
                    for j in range(len(pos_o))
                ]
            )

        ntupels_r2, mc_r2 = raw_corr(
            r_frm,
            r_corr,
            vpar_c._vpar if hasattr(vpar_c, "_vpar") else vpar_c,
            cpar_c._cpar if hasattr(cpar_c, "_cpar") else cpar_c,
            [getattr(c, "_cal", c) for c in cals_c],
        )
        print(f"  RAW with compat-unwrapped: total={mc_r2[3]} match_counts={mc_r2}")

    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
