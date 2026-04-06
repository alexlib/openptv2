"""
Shared fixtures and configuration for engine comparison tests.

This module provides:
- Tolerance values by algorithm module
- Shared test fixtures (synthetic and fixture data)
- Helper functions for both engines
- File-based parameter fixtures: both engines read the SAME files
  through their own readers, testing algorithm parity AND reader parity.
"""

import numpy as np
import pytest
import os
import shutil
from pathlib import Path
from openptv2.test_support import find_test_data_root

FIXTURES = find_test_data_root(Path(__file__))


TOLERANCES = {
    "trafo": 1e-10,
    "calibration": 1e-9,
    "parameters": 1e-9,
    "target": 1e-10,
    "target_array": 1e-10,
    "frame": 1e-10,
    "orientation": 1e-7,
    "imgcoord": 1e-7,
    "correspondences": 1e-7,
    "segmentation": 1e-7,
    "image_processing": 1e-9,
    "epipolar": 1e-7,
    "multimed": 1e-7,
    "tracking_run": 1e-7,
    "tracker": 1e-7,
    "track3d": 1e-5,
}


def get_tolerance(module_name: str) -> float:
    """Get tolerance for a specific module."""
    return TOLERANCES.get(module_name, 1e-7)


# ---------------------------------------------------------------------------
# File paths — the single source of truth for all test data locations
# ---------------------------------------------------------------------------


@pytest.fixture
def calibration_files():
    """Paths to calibration test fixtures."""
    return {
        "cam1": (
            FIXTURES / "calibration" / "cam1.tif.ori",
            FIXTURES / "calibration" / "cam1.tif.addpar",
        ),
        "cam2": (
            FIXTURES / "calibration" / "cam2.tif.ori",
            FIXTURES / "calibration" / "cam2.tif.addpar",
        ),
        "sym_cam1": (
            FIXTURES / "calibration" / "sym_cam1.tif.ori",
            None,
        ),
        "sym_cam2": (
            FIXTURES / "calibration" / "sym_cam2.tif.ori",
            None,
        ),
        "sym_cam3": (
            FIXTURES / "calibration" / "sym_cam3.tif.ori",
            None,
        ),
        "sym_cam4": (
            FIXTURES / "calibration" / "sym_cam4.tif.ori",
            None,
        ),
    }


@pytest.fixture
def control_params_file():
    """Path to control parameters file."""
    return FIXTURES / "control_parameters" / "control.par"


@pytest.fixture
def volume_params_file():
    """Path to volume parameters file."""
    return FIXTURES / "volume_parameters" / "volume.par"


@pytest.fixture
def sequence_params_file():
    """Path to sequence parameters file."""
    return FIXTURES / "sequence_parameters" / "sequence.par"


@pytest.fixture
def tracking_params_file():
    """Path to tracking parameters file."""
    return FIXTURES / "tracking_parameters" / "track.par"


@pytest.fixture
def target_params_file():
    """Path to target parameters file."""
    return FIXTURES / "target_parameters" / "targ_rec.par"


@pytest.fixture
def track_calibration_dir():
    """Path to track calibration directory."""
    return FIXTURES / "track" / "cal"


@pytest.fixture
def test_cavity_dir():
    """Path to test_cavity directory."""
    return FIXTURES / "test_cavity"


@pytest.fixture
def frame_files():
    """Paths to frame test files."""
    return {
        "corres": FIXTURES / "frame" / "rt_is.818",
        "linkage": FIXTURES / "frame" / "ptv_is.818",
    }


# ---------------------------------------------------------------------------
# File-based parameter fixtures — both engines read the SAME files
#
# Each engine uses its own reader (optv: Cython C reader, python: Python
# reader). This tests:
#   1. Algorithm parity (same inputs → same outputs)
#   2. Reader parity (same file → same parameter values)
# ---------------------------------------------------------------------------


def _load_calibrations_from_file(ori_file, addpar_file=None, pos=None, angles=None):
    """Load Calibration objects from files for both engines.

    Each engine reads the same .ori/.addpar files through its own reader.
    Optional pos/angles override after loading.

    Returns:
        (optv_cal, python_cal) tuple
    """
    ori_str = str(ori_file)
    addpar_str = str(addpar_file) if addpar_file else None

    try:
        from optv.calibration import Calibration as OptvCal

        optv_cal = OptvCal()
        optv_cal.from_file(ori_str, addpar_str)
    except Exception:
        optv_cal = None

    try:
        from algorithms.calibration import Calibration as PythonCal

        python_cal = PythonCal.from_file(ori_str, addpar_str)
    except Exception:
        python_cal = None

    if pos is not None:
        if optv_cal is not None:
            optv_cal.set_pos(pos)
        if python_cal is not None:
            python_cal.set_pos(pos)
    if angles is not None:
        if optv_cal is not None:
            optv_cal.set_angles(angles)
        if python_cal is not None:
            python_cal.set_angles(angles)

    return optv_cal, python_cal


def _load_control_params_from_file(par_file):
    """Load ControlParams from file for both engines.

    Returns:
        (optv_cpar, python_cpar) tuple
    """
    par_str = str(par_file)

    try:
        from optv.parameters import ControlParams

        optv_cpar = ControlParams(num_cams=4)
        optv_cpar.read_control_par(par_str)
    except Exception:
        optv_cpar = None

    try:
        from algorithms.parameters import read_control_par

        python_cpar = read_control_par(Path(par_str))
    except Exception:
        python_cpar = None

    return optv_cpar, python_cpar


def _load_volume_params_from_file(par_file):
    """Load VolumeParams from file for both engines.

    Returns:
        (optv_vpar, python_vpar) tuple
    """
    par_str = str(par_file)

    try:
        from optv.parameters import VolumeParams

        optv_vpar = VolumeParams()
        optv_vpar.read_volume_par(par_str)
    except Exception:
        optv_vpar = None

    try:
        from algorithms.parameters import read_volume_par

        python_vpar = read_volume_par(Path(par_str))
    except Exception:
        python_vpar = None

    return optv_vpar, python_vpar


def _load_sequence_params_from_file(par_file):
    """Load SequenceParams from file for both engines.

    Returns:
        (optv_spar, python_spar) tuple
    """
    par_str = str(par_file)

    try:
        from optv.parameters import SequenceParams

        optv_spar = SequenceParams(num_cams=4)
        optv_spar.read_sequence_par(par_str, 4)
    except Exception:
        optv_spar = None

    try:
        from algorithms.parameters import read_sequence_par

        python_spar = read_sequence_par(Path(par_str))
    except Exception:
        python_spar = None

    return optv_spar, python_spar


def _load_tracking_params_from_file(par_file):
    """Load TrackingParams from file for both engines.

    Returns:
        (optv_tpar, python_tpar) tuple
    """
    par_str = str(par_file)

    try:
        from optv.parameters import TrackingParams

        optv_tpar = TrackingParams()
        optv_tpar.read_track_par(par_str)
    except Exception:
        optv_tpar = None

    try:
        from algorithms.parameters import read_track_par

        python_tpar = read_track_par(Path(par_str))
    except Exception:
        python_tpar = None

    return optv_tpar, python_tpar


def _load_target_params_from_file(par_file):
    """Load TargetParams from file for both engines.

    Returns:
        (optv_tpar, python_tpar) tuple
    """
    par_str = str(par_file)

    try:
        from optv.parameters import TargetParams

        optv_tpar = TargetParams()
        optv_tpar.read(par_str)
    except Exception:
        optv_tpar = None

    try:
        from algorithms.parameters import read_target_par

        python_tpar = read_target_par(Path(par_str))
    except Exception:
        python_tpar = None

    return optv_tpar, python_tpar


# ---------------------------------------------------------------------------
# Convenience fixtures — file-based paired parameters
# ---------------------------------------------------------------------------


@pytest.fixture
def file_control_params(control_params_file):
    """ControlParams loaded from file by both engines.

    Returns (optv_cpar, python_cpar) — both read the same file.
    """
    return _load_control_params_from_file(control_params_file)


@pytest.fixture
def file_volume_params(volume_params_file):
    """VolumeParams loaded from file by both engines.

    Returns (optv_vpar, python_vpar) — both read the same file.
    """
    return _load_volume_params_from_file(volume_params_file)


@pytest.fixture
def file_sequence_params(sequence_params_file):
    """SequenceParams loaded from file by both engines.

    Returns (optv_spar, python_spar) — both read the same file.
    """
    return _load_sequence_params_from_file(sequence_params_file)


@pytest.fixture
def file_tracking_params(tracking_params_file):
    """TrackingParams loaded from file by both engines.

    Returns (optv_tpar, python_tpar) — both read the same file.
    """
    return _load_tracking_params_from_file(tracking_params_file)


@pytest.fixture
def file_target_params(target_params_file):
    """TargetParams loaded from file by both engines.

    Returns (optv_tpar, python_tpar) — both read the same file.
    """
    return _load_target_params_from_file(target_params_file)


@pytest.fixture
def file_calibration_cam1(calibration_files):
    """Calibration for cam1 loaded from files by both engines.

    Returns (optv_cal, python_cal) — both read the same .ori/.addpar files.
    """
    ori, addpar = calibration_files["cam1"]
    return _load_calibrations_from_file(ori, addpar)


@pytest.fixture
def file_calibration_4cam(calibration_files):
    """4-camera calibration loaded from files by both engines.

    Returns (optv_cals, python_cals) — lists of 4 calibrations each,
    all loaded from the same sym_cam{1-4}.tif.ori files.
    """
    optv_cals, python_cals = [], []
    for i in range(1, 5):
        ori, addpar = calibration_files[f"sym_cam{i}"]
        o, p = _load_calibrations_from_file(ori, addpar)
        optv_cals.append(o)
        python_cals.append(p)
    return optv_cals, python_cals


# ---------------------------------------------------------------------------
# Synthetic coordinate fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_pixel_coords():
    """Synthetic 2D pixel coordinates for testing."""
    return np.array(
        [
            [100.0, 200.0],
            [300.0, 400.0],
            [500.0, 600.0],
            [1000.0, 500.0],
            [800.0, 300.0],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def synthetic_metric_coords():
    """Synthetic 3D world coordinates for testing."""
    return np.array(
        [
            [10.0, 20.0, 30.0],
            [40.0, 50.0, 60.0],
            [70.0, 80.0, 90.0],
            [100.0, 110.0, 120.0],
            [130.0, 140.0, 150.0],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def synthetic_2d_metric_coords():
    """Synthetic 2D metric coordinates for testing."""
    return np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Legacy helper functions — kept for backward compatibility but marked
# ---------------------------------------------------------------------------


def create_test_target(pnr=0, x=100.0, y=200.0, n=5, nx=2, ny=2, sumg=100.0, tnr=0):
    """Create a test target dictionary."""
    return {
        "pnr": pnr,
        "x": x,
        "y": y,
        "n": n,
        "nx": nx,
        "ny": ny,
        "sumg": sumg,
        "tnr": tnr,
    }


def create_test_target_list(num_targets=10, seed=42):
    """Create a list of test targets with deterministic but varied values."""
    np.random.seed(seed)
    targets = []
    for i in range(num_targets):
        targets.append(
            create_test_target(
                pnr=i,
                x=float(i * 10 + 100),
                y=float(i * 20 + 50),
                n=i + 1,
                nx=max(1, (i % 5) + 1),
                ny=max(1, (i % 3) + 1),
                sumg=float((i + 1) * 100),
                tnr=i % 3,
            )
        )
    return targets


def create_test_control_params():
    """Create test ControlParams for both engines.

    DEPRECATED: Use file_control_params fixture instead, which loads
    both engines' params from the same file.
    """
    try:
        from optv.parameters import ControlParams

        optv_cpar = ControlParams(
            num_cams=4,
            image_size=(1024, 1024),
            pixel_size=(0.01, 0.01),
        )
    except Exception as e:
        print(f"Error creating optv ControlParams: {e}")
        optv_cpar = None

    try:
        from algorithms.parameters import ControlPar

        python_cpar = ControlPar()
        python_cpar.imx = 1024
        python_cpar.imy = 1024
        python_cpar.pix_x = 0.01
        python_cpar.pix_y = 0.01
    except Exception as e:
        print(f"Error creating python ControlPar: {e}")
        python_cpar = None

    return optv_cpar, python_cpar


def create_test_volume_params():
    """Create test VolumeParams for both engines.

    DEPRECATED: Use file_volume_params fixture instead, which loads
    both engines' params from the same file.
    """
    try:
        from optv.parameters import VolumeParams

        optv_vpar = VolumeParams(
            xmin=0.0,
            xmax=100.0,
            ymin=0.0,
            ymax=100.0,
            zmin=0.0,
            zmax=50.0,
        )
    except Exception:
        optv_vpar = None

    try:
        from algorithms.parameters import VolumePar

        python_vpar = VolumePar()
        python_vpar.Xmin = 0.0
        python_vpar.Xmax = 100.0
        python_vpar.Ymin = 0.0
        python_vpar.Ymax = 100.0
        python_vpar.Zmin = 0.0
        python_vpar.Zmax = 50.0
    except Exception:
        python_vpar = None

    return optv_vpar, python_vpar


def create_test_calibration(pos=None, angles=None):
    """Create test Calibration for both engines.

    DEPRECATED: Use file_calibration_cam1 or file_calibration_4cam fixtures
    instead, which load both engines' calibrations from the same files.
    """
    if pos is None:
        pos = np.array([0.0, 0.0, 100.0])
    if angles is None:
        angles = np.array([0.0, 0.0, 0.0])

    try:
        from optv.calibration import Calibration as OptvCal

        optv_cal = OptvCal(pos=pos, angs=angles)
    except Exception:
        optv_cal = None

    try:
        from algorithms.calibration import Calibration as PythonCal

        python_cal = PythonCal()
        python_cal.set_pos(pos)
        python_cal.set_angles(angles)
    except Exception:
        python_cal = None

    return optv_cal, python_cal


def compare_arrays(arr1, arr2, rtol=1e-7, atol=1e-7, name="array"):
    """Helper to compare two arrays with detailed error message."""
    if arr1 is None or arr2 is None:
        return False, f"One or both arrays are None: {name}"
    try:
        np.testing.assert_allclose(arr1, arr2, rtol=rtol, atol=atol)
        return True, "OK"
    except AssertionError as e:
        max_diff = np.max(np.abs(arr1 - arr2)) if arr1.shape == arr2.shape else -1
        return False, f"Max diff: {max_diff:.2e}, shapes: {arr1.shape} vs {arr2.shape}"


def compare_values(val1, val2, rtol=1e-7, atol=1e-7, name="value"):
    """Helper to compare two scalar values."""
    if val1 is None or val2 is None:
        return False, f"One or both values are None: {name}"
    try:
        np.testing.assert_allclose(val1, val2, rtol=rtol, atol=atol)
        return True, "OK"
    except Exception as e:
        return False, f"Difference: {abs(val1 - val2):.2e}"


@pytest.fixture
def num_cameras():
    """Default number of cameras."""
    return 4


@pytest.fixture
def test_image():
    """Create a small test image for segmentation tests."""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[30:35, 40:45] = 200
    img[50:55, 60:65] = 180
    img[70:75, 20:25] = 220
    return img


@pytest.fixture
def test_targets_for_correspondences():
    """Create test target arrays for correspondence tests."""
    np.random.seed(42)
    targets = []
    for i in range(20):
        targets.append(
            create_test_target(
                pnr=i,
                x=float(i * 10 + np.random.rand() * 5),
                y=float(i * 15 + np.random.rand() * 5),
                n=i + 1,
            )
        )
    return targets
