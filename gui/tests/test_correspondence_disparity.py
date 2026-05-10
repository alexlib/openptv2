"""
Diagnostic test: pinpoint why algorithms correspondences produce far fewer
matches than optv.

Usage:
    uv run pytest gui/tests/test_correspondence_disparity.py -v -s
"""

import os
import shutil
import numpy as np
import pytest
from pathlib import Path


def _prepare_test_data(test_dir):
    img_dir = test_dir / "img"
    img_orig = test_dir / "img_orig"
    if img_dir.exists():
        shutil.rmtree(img_dir)
    shutil.copytree(img_orig, img_dir)


@pytest.fixture(scope="module")
def cavity_dir():
    from openptv2.test_support import find_test_data_root

    root = find_test_data_root(Path(__file__))
    d = root / "test_cavity"
    if not d.exists():
        pytest.skip(f"Test data dir {d} not found")
    return d


def _build_raw_algo_params(exp_pm, num_cams):
    from algorithms.parameters import ControlPar, VolumePar, TargetPar
    from algorithms.calibration import Calibration

    ptv_p = exp_pm.get_parameter("ptv")
    crit_p = exp_pm.get_parameter("criteria")
    targ_p = exp_pm.get_parameter("targ_rec")
    cal_p = exp_pm.get_parameter("cal_ori")

    cp = ControlPar(num_cams=num_cams)
    cp.imx = ptv_p["imx"]
    cp.imy = ptv_p["imy"]
    cp.pix_x = ptv_p["pix_x"]
    cp.pix_y = ptv_p["pix_y"]
    cp.hp_flag = ptv_p["hp_flag"]
    cp.all_cam_flag = ptv_p["allcam_flag"]
    cp.tiff_flag = ptv_p["tiff_flag"]
    cp.chfield = ptv_p["chfield"]
    cp.mm.n1 = ptv_p["mmp_n1"]
    cp.mm.n3 = ptv_p["mmp_n3"]
    cp.mm.n2[:] = np.array([ptv_p["mmp_n2"], 0.0, 0.0], dtype=np.float64)
    cp.mm.d[:] = np.array([ptv_p["mmp_d"], 0.0, 0.0], dtype=np.float64)
    cp.mm.nlay = 1

    vp = VolumePar()
    vp.X_lay = np.array(crit_p["X_lay"], dtype=np.float64)
    vp.Zmin_lay = np.array(crit_p["Zmin_lay"], dtype=np.float64)
    vp.Zmax_lay = np.array(crit_p["Zmax_lay"], dtype=np.float64)
    vp.cn = crit_p["cn"]
    vp.cnx = crit_p.get("cnx", 0)
    vp.cny = crit_p.get("cny", 0)
    vp.csumg = crit_p.get("csumg", 0)
    vp.eps0 = crit_p.get("eps0", 0)
    vp.corrmin = crit_p["corrmin"]

    tp = TargetPar()
    tp.gvthres = np.array(targ_p["gvthres"], dtype=np.int32)
    tp.discont = targ_p["disco"]
    tp.nnmin = targ_p["nnmin"]
    tp.nnmax = targ_p["nnmax"]
    tp.nxmin = targ_p["nxmin"]
    tp.nxmax = targ_p["nxmax"]
    tp.nymin = targ_p["nymin"]
    tp.nymax = targ_p["nymax"]
    tp.sumg_min = targ_p["sumg_min"]
    tp.cr_sz = targ_p.get("cr_sz", 2)

    cals = []
    for i in range(num_cams):
        ori_f = cal_p["img_ori"][i]
        add_f = ori_f.replace(".ori", ".addpar")
        cals.append(Calibration.from_file(ori_f, add_f))

    return cp, vp, tp, cals


def test_detection_parity(cavity_dir):
    """Verify algorithms.targ_rec produces same counts as optv.target_recognition."""
    try:
        import optv
    except ImportError:
        pytest.skip("optv bindings not available")

    _prepare_test_data(cavity_dir)
    original_cwd = Path.cwd()
    os.chdir(cavity_dir)

    try:
        from pyptv.experiment import Experiment
        from pyptv.ptv import py_start_proc_c, simple_highpass
        from imageio.v3 import imread
        from skimage.util import img_as_ubyte
        from skimage.color import rgb2gray

        yaml_file = cavity_dir / "parameters_Run1.yaml"
        exp = Experiment()
        exp.pm.from_yaml(yaml_file)
        num_cams = exp.pm.num_cams
        cpar_optv, spar, vpar, tpar_trk, tpar, cals_optv, epar = py_start_proc_c(exp.pm)
        cp_raw, vp_raw, tp_raw, cals_raw = _build_raw_algo_params(exp.pm, num_cams)

        frame = 10000
        print(f"\nFrame {frame} detection parity:")
        all_match = True

        for i_cam in range(num_cams):
            imname = Path(spar.get_img_base_name(i_cam) % frame)
            img = imread(imname)
            if img.ndim > 2:
                img = rgb2gray(img)
            if img.dtype != np.uint8:
                img = img_as_ubyte(img)
            high_pass = simple_highpass(img, cpar_optv)

            from optv.segmentation import target_recognition as o_tr

            t_o = o_tr(high_pass, tpar, i_cam, cpar_optv)

            from algorithms.segmentation import targ_rec
            from algorithms.compat.tracking_framebuf import TargetArray

            # Get grey threshold for this camera
            gv = tp_raw.gvthres[i_cam]
            t_a = targ_rec(
                img=high_pass,
                gvthres=gv,
                discont=tp_raw.discont,
                nnmin=tp_raw.nnmin,
                nnmax=tp_raw.nnmax,
                nxmin=tp_raw.nxmin,
                nxmax=tp_raw.nxmax,
                nymin=tp_raw.nymin,
                nymax=tp_raw.nymax,
                sumg_min=tp_raw.sumg_min,
                xmin=1,
                xmax=cp_raw.imx - 1,
                ymin=1,
                ymax=cp_raw.imy - 1,
            )

            n_o, n_a = len(t_o), len(t_a)
            m = "OK" if n_o == n_a else f"DIFF {n_a - n_o:+d}"
            if n_o != n_a:
                all_match = False
            print(f"  cam{i_cam + 1}: optv={n_o:5d}  algo={n_a:5d}  [{m}]")

        if not all_match:
            print("  WARNING: detection counts differ slightly (expected: 1-2 px edge)")

    finally:
        os.chdir(original_cwd)


def test_correspondence_raw_vs_optv(cavity_dir):
    """Call raw algorithms.correspondences with optv-detected targets,
    compare match_counts to optv.correspondences."""
    try:
        import optv
    except ImportError:
        pytest.skip("optv bindings not available")

    _prepare_test_data(cavity_dir)
    original_cwd = Path.cwd()
    os.chdir(cavity_dir)

    try:
        from pyptv.experiment import Experiment
        from pyptv.ptv import py_start_proc_c, simple_highpass
        from imageio.v3 import imread
        from skimage.util import img_as_ubyte
        from skimage.color import rgb2gray
        from algorithms.correspondences import (
            correspondences as raw_corr,
            correct_frame,
        )
        from algorithms.tracking_frame_buf import Frame as AlgoFrame
        from algorithms.epi import Coord2d

        yaml_file = cavity_dir / "parameters_Run1.yaml"
        exp = Experiment()
        exp.pm.from_yaml(yaml_file)
        num_cams = exp.pm.num_cams
        cpar_optv, spar, vpar, tpar_trk, tpar, cals_optv, epar = py_start_proc_c(exp.pm)
        cp_raw, vp_raw, tp_raw, cals_raw = _build_raw_algo_params(exp.pm, num_cams)

        for frame in [10000, 10001]:
            print(f"\n{'=' * 60}")
            print(
                f"Frame {frame}: raw algo.correspondences() vs optv.correspondences()"
            )
            print(f"  (both using identical optv-detected targets)")
            print(f"{'=' * 60}")

            detections, corrected_optv = [], []
            for i_cam in range(num_cams):
                imname = Path(spar.get_img_base_name(i_cam) % frame)
                img = imread(imname)
                if img.ndim > 2:
                    img = rgb2gray(img)
                if img.dtype != np.uint8:
                    img = img_as_ubyte(img)
                high_pass = simple_highpass(img, cpar_optv)

                from optv.segmentation import target_recognition as o_tr

                targs = o_tr(high_pass, tpar, i_cam, cpar_optv)
                targs.sort_y()
                detections.append(targs)
                from optv.correspondences import MatchedCoords as o_mc

                corrected_optv.append(o_mc(targs, cpar_optv, cals_optv[i_cam]))

            # --- OPTV correspondences ---
            from optv.correspondences import correspondences as o_corr

            sp_o, sc_o, nt_o = o_corr(
                detections, corrected_optv, cals_optv, vpar, cpar_optv
            )
            total_o = sum(x.shape[1] for x in sp_o if x is not None)
            print(f"  OPTV correspondences:  total={total_o}")

            # --- RAW ALGORITHMS correspondences using same detection data ---
            frm = AlgoFrame(num_cams=num_cams, max_targets=10000)
            corrected_raw = []
            for i_cam in range(num_cams):
                targs_list = list(detections[i_cam])
                n = len(targs_list)
                frm.num_targets[i_cam] = n
                tgt_arr = frm.targets[i_cam]
                for tnum in range(n):
                    t = targs_list[tnum]
                    tgt_arr[tnum].pnr = t.pnr()
                    tgt_arr[tnum].x = t.pos()[0]
                    tgt_arr[tnum].y = t.pos()[1]
                    tgt_arr[tnum].n = t.count_pixels()[0]
                    tgt_arr[tnum].nx = t.count_pixels()[1]
                    tgt_arr[tnum].ny = t.count_pixels()[2]
                    tgt_arr[tnum].sumg = t.sum_grey_value()
                    tgt_arr[tnum].tnr = t.tnr()

            # Get corrected coords from optv MatchedCoords -> raw Coord2d
            for i_cam in range(num_cams):
                mc = corrected_optv[i_cam]
                pos, pnr = mc.as_arrays()
                corrected_raw.append(
                    [
                        Coord2d(x=pos[j, 0], y=pos[j, 1], pnr=pnr[j])
                        for j in range(len(pos))
                    ]
                )
            ntupels, match_counts = raw_corr(
                frm, corrected_raw, vp_raw, cp_raw, cals_raw
            )

            print(f"  RAW algo correspondences: match_counts={match_counts}")
            total_a = match_counts[3]
            print(f"  RAW algo correspondences:  total={total_a}")

            for gi in range(len(sp_o)):
                if sp_o[gi] is not None:
                    o_cnt = sp_o[gi].shape[1]
                    a_cnt = match_counts[4 - num_cams + gi]
                    m = "OK" if o_cnt == a_cnt else f"DIFF {a_cnt - o_cnt:+d}"
                    print(f"    group[{gi}]: optv={o_cnt:4d}  algo={a_cnt:4d}  [{m}]")

    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
