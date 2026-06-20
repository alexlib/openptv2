import sys
import importlib.util
from pathlib import Path
import numpy as np
import pytest

from algorithms.calibration import Calibration as PyCalibration
from algorithms.parameters import MmNp as PyMmNp
from algorithms.imgcoord import (
    img_coord as py_img_coord,
    flat_image_coord as py_flat_image_coord,
    img_coord_batch as py_img_coord_batch,
    flat_image_coord_batch as py_flat_image_coord_batch
)

# Try to import legacy optv bindings
try:
    from optv.calibration import Calibration as CCalib
    from optv.parameters import MultimediaParams as CMultiParams
    from optv.imgcoord import (
        image_coordinates as c_img_coord,
        flat_image_coordinates as c_flat_img_coord
    )
    HAS_OPTV = True
except ImportError:
    HAS_OPTV = False


# Helper to load interpreted/fallback version of algorithms/imgcoord.py
def load_interpreted_imgcoord():
    path = Path(__file__).parent.parent / "imgcoord.py"
    spec = importlib.util.spec_from_file_location("algorithms.imgcoord_fallback", str(path))
    module = importlib.util.module_from_spec(spec)
    module.__package__ = "algorithms"
    # Ensure a clean module namespace
    sys.modules["algorithms.imgcoord_fallback"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def interpreted_module():
    return load_interpreted_imgcoord()


def test_compiled_vs_interpreted_flags(interpreted_module):
    """Verify that we are indeed comparing a compiled module against an interpreted fallback."""
    import algorithms.imgcoord as imgcoord_compiled
    
    print("Compiled module is_compiled():", imgcoord_compiled.is_compiled())
    print("Interpreted module is_compiled():", interpreted_module.is_compiled())
    
    # Assert compile states
    assert imgcoord_compiled.is_compiled() is True
    assert interpreted_module.is_compiled() is False


@pytest.mark.skipif(not HAS_OPTV, reason="Legacy optv bindings not available")
def test_imgcoord_numerical_parity(interpreted_module):
    """Verify numerical parity between legacy optv, compiled algorithms, and interpreted fallback.

    Uses tolerance of 1e-7.
    """
    # Calibration files from test_data
    test_data_dir = Path(__file__).parent.parent.parent / "test_data" / "synthetic"
    ori_file = test_data_dir / "cal" / "cam1.tif.ori"
    add_file = test_data_dir / "cal" / "cam1.tif.addpar"
    
    # 1. Load legacy C/Cython objects
    c_cal = CCalib()
    c_cal.from_file(ori_file=str(ori_file), add_file=str(add_file))
    
    c_mm = CMultiParams(n1=1.0, n3=1.0)
    c_mm.set_layers(np.array([1.0]), np.array([1.0]))
    
    # 2. Load modern python/compiled objects
    py_cal = PyCalibration.from_file(ori_file, add_file)
    py_mm = PyMmNp(nlay=1, n1=1.0, n2=[1.0], d=[1.0], n3=1.0)
    
    # 3. Define test 3D points
    points = np.array([
        [10.0, 15.0, 100.0],
        [-20.0, 5.0, 120.0],
        [0.0, 0.0, 110.0],
        [50.0, -30.0, 80.0]
    ], dtype=np.float64)
    
    # --- Compare batch projection (distorted) ---
    c_dist_res = c_img_coord(points, c_cal, c_mm)
    
    # Compiled algorithms imgcoord batch
    py_dist_res_compiled = py_img_coord_batch(points, py_cal, py_mm)
    
    # Interpreted fallback algorithms imgcoord batch
    py_dist_res_interpreted = interpreted_module.img_coord_batch(points, py_cal, py_mm)
    
    # Verify distorted parity
    np.testing.assert_allclose(py_dist_res_compiled, c_dist_res, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(py_dist_res_interpreted, c_dist_res, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(py_dist_res_compiled, py_dist_res_interpreted, rtol=1e-7, atol=1e-7)
    
    # --- Compare batch projection (flat) ---
    c_flat_res = c_flat_img_coord(points, c_cal, c_mm)
    
    # Compiled algorithms flat imgcoord batch
    py_flat_res_compiled = py_flat_image_coord_batch(points, py_cal, py_mm)
    
    # Interpreted fallback algorithms flat imgcoord batch
    py_flat_res_interpreted = interpreted_module.flat_image_coord_batch(points, py_cal, py_mm)
    
    # Verify flat parity
    np.testing.assert_allclose(py_flat_res_compiled, c_flat_res, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(py_flat_res_interpreted, c_flat_res, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(py_flat_res_compiled, py_flat_res_interpreted, rtol=1e-7, atol=1e-7)


def test_scalar_parity_compiled_vs_interpreted(interpreted_module):
    """Verify that scalar single-point projection functions match perfectly."""
    # Build a custom mock calibration
    cal = PyCalibration()
    cal.ext_par.x0 = 0.0
    cal.ext_par.y0 = 0.0
    cal.ext_par.z0 = 50.0
    cal.ext_par.dm = np.eye(3, dtype=np.float64)
    cal.int_par.cc = 12.0
    cal.int_par.xh = 0.05
    cal.int_par.yh = -0.05
    cal.glass_par.vec_x = 0.0
    cal.glass_par.vec_y = 0.0
    cal.glass_par.vec_z = 25.0
    cal.added_par.k1 = -0.002
    cal.added_par.k2 = 0.0
    cal.added_par.k3 = 0.0
    
    mm = PyMmNp(nlay=1, n1=1.0, n2=[1.0], d=[1.0], n3=1.0)
    
    pos = np.array([10.0, 5.0, -10.0], dtype=np.float64)
    
    # Compiled scalar flat
    flat_x_c, flat_y_compiled = py_flat_image_coord(
        pos,
        cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0, cal.ext_par.dm, cal.int_par.cc,
        cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z,
        mm.n1, mm.n2[0], mm.n3, mm.d[0]
    )
    
    # Interpreted scalar flat
    flat_x_i, flat_y_interpreted = interpreted_module.flat_image_coord(
        pos,
        cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0, cal.ext_par.dm, cal.int_par.cc,
        cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z,
        mm.n1, mm.n2[0], mm.n3, mm.d[0]
    )
    
    assert abs(flat_x_c - flat_x_i) < 1e-12
    assert abs(flat_y_compiled - flat_y_interpreted) < 1e-12
    
    # Compiled scalar distorted
    dist_x_c, dist_y_compiled = py_img_coord(pos, cal, mm)
    
    # Interpreted scalar distorted
    dist_x_i, dist_y_interpreted = interpreted_module.img_coord(pos, cal, mm)
    
    assert abs(dist_x_c - dist_x_i) < 1e-12
    assert abs(dist_y_compiled - dist_y_interpreted) < 1e-12
