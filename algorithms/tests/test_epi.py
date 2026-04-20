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
    print(f"epi_mm_2d: out={out}")
    assert np.abs(out[0] - 0.85858163) < EPS
    assert np.abs(out[1] - 8.58581626) < EPS
    assert np.abs(out[2] - 0.0) < EPS

    # Trivial case
    out = epi_mm_2d(0.0, 0.0, cal, mm_n1, mm_n2_0, mm_n3, mm_d0, vpar_X_lay, vpar_Zmin_lay, vpar_Zmax_lay)
    print(f"epi_mm_2d: out={out}")
    assert np.abs(out[0] - 0.0) < EPS
    assert np.abs(out[1] - 0.0) < EPS
    assert np.abs(out[2] - 0.0) < EPS

def test_epi_mm():
    cal1 = {
        "dm": np.eye(3),
        "x0": 10.0,
        "y0": 0.0,
        "z0": 100.0,
        "cc": 100.0,
        "gx": 0.0,
        "gy": 0.0,
        "gz": 50.0,
    }
    cal2 = {
        "dm": np.eye(3),
        "x0": -10.0,
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
    vpar_Zmin_lay = (-50.0, -50.0)
    vpar_Zmax_lay = (50.0, 50.0)
    x = 10.0
    y = 10.0
    xmin, ymin, xmax, ymax = epi_mm(x, y, cal1, cal2, mm_n1, mm_n2_0, mm_n3, mm_d0, vpar_X_lay, vpar_Zmin_lay, vpar_Zmax_lay)
    print(f"epi_mm: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")
    assert np.abs(xmin - 26.44927852) < EPS
    assert np.abs(xmax - 51.60078764) < EPS
    assert np.abs(ymin - 10.08218486) < EPS
    assert np.abs(ymax - 10.04378909) < EPS

def test_epi_mm_perpendicular():
    cal1 = {
        "dm": np.eye(3),
        "x0": 0.0,
        "y0": 0.0,
        "z0": 100.0,
        "cc": 100.0,
        "gx": 0.0,
        "gy": 0.0,
        "gz": 50.0,
    }
    cal2 = {
        "dm": np.eye(3),
        "x0": 100.0,
        "y0": 0.0,
        "z0": 0.0,
        "cc": 100.0,
        "gx": 0.0,
        "gy": 0.0,
        "gz": 50.0,
    }
    mm_n1 = 1.0
    mm_n2_0 = 1.0
    mm_n3 = 1.0
    mm_d0 = 1.0
    vpar_X_lay = (-100.0, 100.0)
    vpar_Zmin_lay = (-100.0, -100.0)
    vpar_Zmax_lay = (100.0, 100.0)
    x = 0.0
    y = 0.0
    xmin, ymin, xmax, ymax = epi_mm(x, y, cal1, cal2, mm_n1, mm_n2_0, mm_n3, mm_d0, vpar_X_lay, vpar_Zmin_lay, vpar_Zmax_lay)
    print(f"epi_mm_perpendicular: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}")
    assert np.abs(xmin + 100.0) < EPS
    assert np.abs(xmax - 100.0) < EPS
    assert np.abs(ymin - 0.0) < EPS
    assert np.abs(ymax - 0.0) < EPS

# Placeholder for test_find_candidate (requires more context about the Python API and data structures)
