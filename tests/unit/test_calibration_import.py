"""Tests for openptv2.calibration_import."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from openptv2.algorithms.calibration import Calibration
from openptv2.calibration_import import (
    calibration_from_opencv,
    opencv_from_calibration,
    read_opencv_flat15,
    similarity_from_correspondences,
)


@pytest.mark.unit
def test_calibration_from_opencv_roundtrip():
    """Verify calibration_from_opencv and opencv_from_calibration roundtrip."""
    fx, fy = 7000.0, 7000.0
    cx, cy = 1280.0, 1024.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
    dist = np.array([1e-4, -2e-8, 0.0, 0.0, 0.0])

    rvec = np.array([0.1, -0.2, 0.3])
    tvec = np.array([100.0, -200.0, 3000.0])

    imx, imy = 2560, 2048
    pix_x = 0.005

    cal, pix_y = calibration_from_opencv(K, dist, rvec, tvec, imx=imx, imy=imy, pix_x=pix_x)
    assert isinstance(cal, Calibration)
    assert cal.int_par.cc == pytest.approx(fx * pix_x)

    # Check roundtrip
    K_rec, dist_rec, rvec_rec, tvec_rec = opencv_from_calibration(cal, imx=imx, imy=imy, pix_x=pix_x, pix_y=pix_y)
    assert np.allclose(K, K_rec, atol=1e-6)
    assert np.allclose(rvec, rvec_rec, atol=1e-6)
    assert np.allclose(tvec, tvec_rec, atol=1e-6)


@pytest.mark.unit
def test_similarity_from_correspondences():
    """Verify Kabsch alignment recovers rotation and translation."""
    rng = np.random.default_rng(123)
    pts_src = rng.uniform(-100, 100, (20, 3))

    R_true = Rotation.from_euler("xyz", [15, -20, 35], degrees=True).as_matrix()
    t_true = np.array([50.0, -30.0, 120.0])

    pts_dst = (R_true @ pts_src.T).T + t_true

    A_est, b_est, s_est = similarity_from_correspondences(pts_src, pts_dst)

    assert np.allclose(R_true, A_est, atol=1e-8)
    assert np.allclose(t_true, b_est, atol=1e-8)
    assert s_est == pytest.approx(1.0)
