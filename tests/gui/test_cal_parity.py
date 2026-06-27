"""
Debug: compare calibration/distortion parameters between optv and compat.

Usage:
    uv run pytest tests/gui/test_cal_parity.py -v -s
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


def test_calibration_parameter_parity(cavity_dir):
    try:
        import optv
    except ImportError:
        pytest.skip("optv bindings not available")

    _prepare_test_data(cavity_dir)
    original_cwd = Path.cwd()
    os.chdir(cavity_dir)

    try:
        from openptv2.gui.experiment import Experiment
        from openptv2.gui.ptv import py_start_proc_c

        yaml_file = cavity_dir / "parameters_Run1.yaml"
        exp = Experiment()
        exp.pm.from_yaml(yaml_file)
        num_cams = exp.pm.num_cams
        cal_p = exp.pm.get_parameter("cal_ori")

        cpar, spar, vpar, trk, tpar, cals_optv, epar = py_start_proc_c(exp.pm)

        from openptv2.algorithms.compat.calibration import Calibration as CalC

        print("\nCalibration parameter comparison (optv vs compat):")

        for i in range(num_cams):
            cal_o = cals_optv[i]

            cal_c = CalC()
            ori_f = cal_p["img_ori"][i]
            add_f = ori_f.replace(".ori", ".addpar")
            cal_c.from_file(ori_f, add_f)

            rc = cal_c._cal
            pos_o = cal_o.get_pos()
            pos_c = cal_c.get_pos()
            print(f"\n  cam{i + 1}:")
            print(f"    pos:        optv={pos_o}  compat={pos_c}")
            ang_o = cal_o.get_angles()
            ang_c = cal_c.get_angles()
            print(f"    angles:     optv={ang_o}  compat={ang_c}")
            pp_o = cal_o.get_primary_point()
            pp_c = cal_c.get_primary_point()
            print(f"    pri_pt:     optv={pp_o}  compat={pp_c}")
            rd_o = cal_o.get_radial_distortion()
            rd_c = cal_c.get_radial_distortion()
            print(f"    rad_dist:   optv={rd_o}  compat={rd_c}")
            dc_o = cal_o.get_decentering()
            dc_c = cal_c.get_decentering()
            print(f"    decent:     optv={dc_o}  compat={dc_c}")
            af_o = cal_o.get_affine()
            af_c = cal_c.get_affine()
            print(f"    affine:     optv={af_o}  compat={af_c}")
            gv_o = cal_o.get_glass_vec()
            gv_c = cal_c.get_glass_vec()
            print(f"    glass:      optv={gv_o}  compat={gv_c}")

            # Also check pixel to metric conversion
            from openptv2.algorithms.compat.transforms import convert_arr_pixel_to_metric

            test_pts = np.array([[100.0, 200.0], [500.0, 300.0]], dtype=np.float64)

            # optv pixel_to_metric
            metric_o = np.empty_like(test_pts)
            for j in range(len(test_pts)):
                xo, yo = test_pts[j]
                # optv transforms module
                from optv.transforms import convert_arr_pixel_to_metric as o_ptm

            cpar_c = cpar.control_par if hasattr(cpar, 'control_par') else cpar
            metric_c = convert_arr_pixel_to_metric(test_pts, cpar_c)

            print(
                f"    px->metric test pts={test_pts[0]} -> "
                f"opTV=({metric_o[0, 0]:.4f},{metric_o[0, 1]:.4f}) "
                f"compat=({metric_c[0, 0]:.4f},{metric_c[0, 1]:.4f})"
            )

    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
