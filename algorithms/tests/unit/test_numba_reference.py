"""Numba stability and libopt-style reference checks.

This suite complements compile smoke tests by asserting:
1) no hard crashes during warmup/compilation,
2) numeric behavior close to C reference unit tests,
3) execution on real calibration/parameter files.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


class TestNumbaStability:
    """Crash-focused checks for JIT warmup and compilation."""

    def test_warmup_compiles_all_registered_functions(self):
        from algorithms.tests.conftest_numba_warmup import _warmup_all

        count, elapsed = _warmup_all()
        assert count == 47
        assert elapsed >= 0.0

    def test_warmup_module_runs_in_subprocess_without_crash(self, tmp_path):
        cache_dir = tmp_path / "numba_cache"
        env = os.environ.copy()
        env["NUMBA_CACHE_DIR"] = str(cache_dir)

        proc = subprocess.run(
            [sys.executable, "-m", "algorithms.tests.conftest_numba_warmup"],
            cwd=Path(__file__).resolve().parents[3],
            env=env,
            capture_output=True,
            text=True,
        )

        assert proc.returncode == 0, (
            "Numba warmup crashed in subprocess.\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )


class TestLiboptReferenceValues:
    """Reference-value checks aligned with lib/tests/check_*.c expectations."""

    def test_vec_utils_reference_values(self):
        from algorithms.vec_utils import vec_cross, vec_diff_norm, vec_dot

        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 2.0, 0.0])
        np.testing.assert_allclose(vec_dot(a, b), 0.0, atol=1e-12)

        c = np.array([2.0, 2.0, 0.0])
        np.testing.assert_allclose(vec_dot(c, a), 2.0, atol=1e-12)

        out = vec_cross(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        np.testing.assert_allclose(out, np.array([0.0, 0.0, 1.0]), atol=1e-12)

        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([4.0, 5.0, 6.0])
        np.testing.assert_allclose(vec_diff_norm(v1, v2), np.sqrt(3.0) * 3.0, atol=1e-12)

    def test_trafo_reference_values(self):
        from algorithms.trafo import fast_metric_to_pixel, fast_pixel_to_metric

        xp, yp = fast_metric_to_pixel(0.0, 0.0, 1024, 1008, 0.01, 0.01)
        np.testing.assert_allclose([xp, yp], [512.0, 504.0], atol=1e-12)

        xp, yp = fast_metric_to_pixel(1.0, 0.0, 1024, 1008, 0.01, 0.01)
        np.testing.assert_allclose([xp, yp], [612.0, 504.0], atol=1e-12)

        xp, yp = fast_metric_to_pixel(0.0, -1.0, 1024, 1008, 0.01, 0.01)
        np.testing.assert_allclose([xp, yp], [512.0, 604.0], atol=1e-12)

        xm, ym = fast_pixel_to_metric(512.0, 504.0, 1024, 1008, 0.01, 0.01)
        np.testing.assert_allclose([xm, ym], [0.0, 0.0], atol=1e-12)

    def test_track_reference_values(self):
        from algorithms.parameters import TrackParTuple
        from algorithms.track import angle_acc, pos3d_in_bounds, search_volume_center_moving

        prev_pos = np.array([1.1, 0.6, 0.1])
        curr_pos = np.array([2.0, -0.8, 0.2])
        center = search_volume_center_moving(prev_pos, curr_pos)
        np.testing.assert_allclose(center, np.array([2.9, -2.2, 0.3]), atol=1e-12)

        bounds = TrackParTuple(
            dvxmin=-2.0,
            dvxmax=2.0,
            dvymin=-2.0,
            dvymax=2.0,
            dvzmin=-2.0,
            dvzmax=2.0,
            dangle=120.0,
            dacc=0.4,
            add=0,
            dsumg=0.0,
            dn=0.0,
            dnx=0.0,
            dny=0.0,
        )
        assert pos3d_in_bounds(np.array([1.0, -1.0, 0.0]), bounds) is True
        assert pos3d_in_bounds(np.array([2.0, -0.8, 2.1]), bounds) is False

        angle, acc = angle_acc(
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([1.1, 1.0, 1.0]),
        )
        np.testing.assert_allclose(angle, 2.902234, atol=1e-5)
        np.testing.assert_allclose(acc, 0.1, atol=1e-10)

    def test_ray_tracing_reference_values(self):
        from algorithms.ray_tracing import fast_ray_tracing

        dm = np.array(
            [
                [1.0, 0.2, -0.3],
                [0.2, 1.0, 0.0],
                [-0.3, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        x, out = fast_ray_tracing(
            np.array([100.0, 100.0, -100.0], dtype=np.float64),
            dm,
            np.array([0.0, 0.0, 100.0], dtype=np.float64),
            np.array([0.0001, 0.00001, 1.0], dtype=np.float64),
            5.0,
            1.0,
            1.49,
            1.33,
        )

        np.testing.assert_allclose(x, np.array([110.406944, 88.325788, 0.988076]), atol=2e-4)
        np.testing.assert_allclose(out, np.array([0.387960, 0.310405, -0.867834]), atol=2e-4)


class TestRealDataNumbaExecution:
    """Run compiled paths against repo fixture files to catch integration regressions."""

    def test_real_data_projection_pipeline(
        self,
        calibration_files,
        control_params_file,
        volume_params_file,
    ):
        from algorithms.calibration import Calibration
        from algorithms.multimed import CalibRawArrays, init_mmlut
        from algorithms.parameters import read_control_par, read_volume_par

        ori, add = calibration_files["cam1"]
        cal = Calibration()
        cal.from_file(ori, add)

        cpar = read_control_par(Path(control_params_file))
        vpar = read_volume_par(Path(volume_params_file))

        cal = init_mmlut(vpar, cpar, cal)
        raw = CalibRawArrays(cal, cpar)

        px, py = raw.project(np.array([1.0, 1.0, -1.0], dtype=np.float64))
        assert np.isfinite(px)
        assert np.isfinite(py)

    def test_real_data_epi_endpoints(
        self,
        calibration_files,
        control_params_file,
        volume_params_file,
    ):
        from algorithms.calibration import Calibration
        from algorithms.epi import epi_mm
        from algorithms.multimed import init_mmlut
        from algorithms.parameters import read_control_par, read_volume_par

        ori1, add1 = calibration_files["cam1"]
        ori2, add2 = calibration_files["sym_cam2"]

        cal1 = Calibration().from_file(ori1, add1)
        cal2 = Calibration().from_file(ori2, add2)

        cpar = read_control_par(Path(control_params_file))
        vpar = read_volume_par(Path(volume_params_file))

        cal1 = init_mmlut(vpar, cpar, cal1)
        cal2 = init_mmlut(vpar, cpar, cal2)

        xmin, xmax, ymin, ymax = epi_mm(1.0, 1.0, cal1, cal2, cpar.mm, vpar)

        assert np.isfinite(xmin)
        assert np.isfinite(xmax)
        assert np.isfinite(ymin)
        assert np.isfinite(ymax)

    def test_real_data_ray_tracing(self, calibration_files, control_params_file):
        from algorithms.calibration import Calibration
        from algorithms.parameters import read_control_par
        from algorithms.ray_tracing import ray_tracing

        ori, add = calibration_files["cam1"]
        cal = Calibration().from_file(ori, add)
        cpar = read_control_par(Path(control_params_file))

        x, out = ray_tracing(1.0, 1.0, cal, cpar.mm)

        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(out))
