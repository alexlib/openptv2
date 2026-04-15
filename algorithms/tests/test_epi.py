import numpy as np
import pytest

EPS = 1e-5

from algorithms.epi import epi_mm_2d, epi_mm, find_candidate

def test_epi_mm_2d():
    # Setup as in check_epi.c
    cal = {
        "dm": np.eye(3),
        "x0": 0.0,
        "y0": 0.0,
        "z0": 100.0,
        "cc": 100.0,
        "gx": 0.0,
        "gy": 0.0,
        "gz": 50.0,
    }
    mm_n1 = 1.0
    mm_n2_0 = 1.49
    mm_n3 = 1.33
    mm_d0 = 5.0
    vpar_X_lay = (-250.0, 250.0)
    vpar_Zmin_lay = (-100.0, -100.0)
    vpar_Zmax_lay = (100.0, 100.0)

    # Non-trivial case
    out = epi_mm_2d(1.0, 10.0, cal, mm_n1, mm_n2_0, mm_n3, mm_d0, vpar_X_lay, vpar_Zmin_lay, vpar_Zmax_lay)
    assert np.abs(out[0] - 0.85858163) < EPS
    assert np.abs(out[1] - 8.58581626) < EPS
    assert np.abs(out[2] - 0.0) < EPS

    # Trivial case
    out = epi_mm_2d(0.0, 0.0, cal, mm_n1, mm_n2_0, mm_n3, mm_d0, vpar_X_lay, vpar_Zmin_lay, vpar_Zmax_lay)
    assert np.abs(out[0] - 0.0) < EPS
    assert np.abs(out[1] - 0.0) < EPS
    assert np.abs(out[2] - 0.0) < EPS

# Additional tests for epi_mm, epi_mm_perpendicular, and find_candidate would follow the same pattern,
# using the same parameter values and assertions as in check_epi.c.
