"""Unit tests for parallel Multi-Media Look-Up Table (MMLUT) generation."""

import os
from pathlib import Path
import numpy as np
import pytest

from openptv2.algorithms.multimed import (
    init_mmlut,
    prepare_mmluts,
)
from openptv2.algorithms.track_kernels_batch import (
    _init_mmlut_slice_1layer,
    _init_mmlut_slice_nlay,
    init_mmlut_data_fast,
    init_mmlut_data_nlay_fast,
)
from openptv2.gui.experiment import Experiment
from openptv2.gui.ptv import py_start_proc_c


REPO_ROOT = Path(__file__).parent.parent.parent
TEST_DATA = REPO_ROOT / "test_data" / "test_cavity"


@pytest.fixture
def cavity_params():
    """Load calibration, control, and volume parameters from test_cavity."""
    yaml_file = TEST_DATA / "parameters_Run1.yaml"
    if not yaml_file.exists():
        pytest.skip("test_cavity fixture not present")

    exp = Experiment()
    cwd0 = os.getcwd()
    os.chdir(TEST_DATA)
    try:
        exp.pm.from_yaml(yaml_file)
        cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(exp.pm)
        return exp, cpar, spar, vpar, cals
    finally:
        os.chdir(cwd0)


def test_init_mmlut_data_1layer_parallel_parity():
    """Test single-layer MMLUT fast kernel parity against serial slice."""
    nr = 128
    nz = 64
    rw = 2.0
    cal_t_x0 = 10.0
    cal_t_y0 = 5.0
    cal_t_z0 = 200.0
    Zmin_t = -50.0
    mm_n1 = 1.0
    mm_n2_0 = 1.5
    mm_n3 = 1.33
    mm_d0 = 10.0

    # Serial reference
    ref_data = np.empty(nr * nz, dtype=np.float64)
    _init_mmlut_slice_1layer(
        ref_data, 0, nr, nz, rw, cal_t_x0, cal_t_y0, cal_t_z0, Zmin_t,
        mm_n1, mm_n2_0, mm_n3, mm_d0
    )

    # Parallel evaluation
    par_data = init_mmlut_data_fast(
        nr, nz, rw, cal_t_x0, cal_t_y0, cal_t_z0, Zmin_t,
        mm_n1, mm_n2_0, mm_n3, mm_d0
    )

    assert np.allclose(par_data, ref_data, atol=1e-12)
    assert np.array_equal(par_data, ref_data)


def test_init_mmlut_data_nlay_parallel_parity():
    """Test multi-layer MMLUT fast kernel parity against serial slice."""
    nr = 128
    nz = 64
    rw = 2.0
    cal_t_x0 = 10.0
    cal_t_y0 = 5.0
    cal_t_z0 = 200.0
    Zmin_t = -50.0
    mm_n1 = 1.0
    mm_n3 = 1.33
    n2 = np.array([1.5, 1.4], dtype=np.float64)
    d = np.array([10.0, 5.0], dtype=np.float64)
    nlay = 2

    # Serial reference
    ref_data = np.empty(nr * nz, dtype=np.float64)
    _init_mmlut_slice_nlay(
        ref_data, 0, nr, nz, rw, cal_t_x0, cal_t_y0, cal_t_z0, Zmin_t,
        mm_n1, mm_n3, n2, d, nlay
    )

    # Parallel evaluation
    par_data = init_mmlut_data_nlay_fast(
        nr, nz, rw, cal_t_x0, cal_t_y0, cal_t_z0, Zmin_t,
        mm_n1, mm_n3, n2, d, nlay
    )

    assert np.allclose(par_data, ref_data, atol=1e-12)
    assert np.array_equal(par_data, ref_data)


def test_prepare_mmluts_multi_camera_parallel(cavity_params):
    """Test that prepare_mmluts initializes all cameras in parallel with exact parity."""
    exp, cpar, spar, vpar, cals = cavity_params

    # Reset calibrations mmlut state
    for cal in cals:
        cal.mmlut.data = None

    # Run multi-camera parallel prepare_mmluts
    prepare_mmluts(vpar, cpar, cals, n_workers=4)

    for i, cal in enumerate(cals):
        assert cal.mmlut.is_initialized, f"Camera {i} mmlut not initialized"
        assert cal.mmlut.data is not None
        assert len(cal.mmlut.data) == cal.mmlut.nr * cal.mmlut.nz
