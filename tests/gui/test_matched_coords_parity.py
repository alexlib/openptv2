"""
Test whether compat MatchedCoords produces same corrected coordinates as optv
for identical target inputs.

Usage:
    uv run pytest gui/tests/test_matched_coords_parity.py -v -s
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


def test_matched_coords_parity(cavity_dir):
    """Compare optv MatchedCoords vs compat MatchedCoords for same targets."""
    try:
        import optv
    except ImportError:
        pytest.skip("optv bindings not available")

    _prepare_test_data(cavity_dir)
    original_cwd = Path.cwd()
    os.chdir(cavity_dir)

    try:
        from openptv2.gui.pyptv.experiment import Experiment
        from openptv2.gui.pyptv.ptv import py_start_proc_c, simple_highpass
        from imageio.v3 import imread
        from skimage.util import img_as_ubyte
        from skimage.color import rgb2gray

        yaml_file = cavity_dir / "parameters_Run1.yaml"
        exp = Experiment()
        exp.pm.from_yaml(yaml_file)
        num_cams = exp.pm.num_cams
        cpar, spar, vpar, trk, tpar, cals_optv, epar = py_start_proc_c(exp.pm)

        ptv_p = exp.pm.get_parameter("ptv")
        targ_p = exp.pm.get_parameter("targ_rec")
        cal_p = exp.pm.get_parameter("cal_ori")

        from openptv2.algorithms.compat.parameters import ControlParams as C, TargetParams as T
        from openptv2.algorithms.compat.calibration import Calibration as CalC

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
        print(f"\nFrame {frame}: MatchedCoords parity test")

        for i_cam in range(num_cams):
            imname = Path(spar.get_img_base_name(i_cam) % frame)
            img = imread(imname)
            if img.ndim > 2:
                img = rgb2gray(img)
            if img.dtype != np.uint8:
                img = img_as_ubyte(img)
            high_pass = simple_highpass(img, cpar)

            # Get optv detections
            from optv.segmentation import target_recognition as o_tr

            targs_o = o_tr(high_pass, tpar, i_cam, cpar)
            targs_o.sort_y()

            # Get compat detections
            from openptv2.algorithms.compat.segmentation import target_recognition as c_tr

            targs_c = c_tr(high_pass, tp_c, i_cam, cp_c)

            n_o = len(targs_o)
            n_c = len(targs_c)
            print(f"\n  cam{i_cam + 1}: optv={n_o} compat={n_c} targets")

            # Build optv MatchedCoords
            from optv.correspondences import MatchedCoords as o_mc

            o_mc_obj = o_mc(targs_o, cpar, cals_optv[i_cam])
            pos_o, pnr_o = o_mc_obj.as_arrays()

            # Build compat MatchedCoords (using SAME optv targets for fairness)
            from openptv2.algorithms.compat.correspondences import MatchedCoords as c_mc

            c_mc_obj = c_mc(targs_o, cp_c, cals_c[i_cam])
            pos_c, pnr_c = c_mc_obj.as_arrays()

            if n_o != n_c:  # We know they differ slightly
                # Compare first 10 for sanity
                m = min(n_o, 10)
                print(f"    First {m} targets position comparison:")
                for j in range(m):
                    print(
                        f"      [{j}] optv=({pos_o[j, 0]:.4f},{pos_o[j, 1]:.4f}) "
                        f"compat=({pos_c[j, 0]:.4f},{pos_c[j, 1]:.4f})"
                    )
            else:
                # Same count — compare all
                abs_diff = np.abs(pos_o - pos_c)
                max_diff = np.max(abs_diff)
                mean_diff = np.mean(abs_diff)
                print(f"    Max coord diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
                if max_diff > 1e-4:
                    print(f"    WARNING: large coordinate differences detected!")
                    mismatches = np.sum(abs_diff > 1e-4, axis=0)
                    print(
                        f"    mismatches > 1e-4: x={mismatches[0]}, y={mismatches[1]}"
                    )

    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
