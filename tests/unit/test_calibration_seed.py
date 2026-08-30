"""Tests for openptv2.calibration_seed."""

from pathlib import Path
import numpy as np
import pytest

from openptv2.algorithms.calibration import Calibration, Exterior, Interior, AddedPar, Glass
from openptv2.algorithms.parameters import ControlPar, MmNp
from openptv2.calibration_seed import (
    angles_from_dm,
    exterior_from_rotation,
    dm_from_lookat,
    seed_from_lookat,
    seed_from_dlt,
    read_rig,
    seed_rig,
)


@pytest.mark.unit
def test_angles_roundtrip():
    """Verify angles_from_dm inverts compute_rotation_matrix."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        omega = float(rng.uniform(-np.pi, np.pi))
        phi = float(rng.uniform(-np.pi / 2.5, np.pi / 2.5))
        kappa = float(rng.uniform(-np.pi, np.pi))
        
        ext = Exterior(x0=0.0, y0=0.0, z0=0.0, omega=omega, phi=phi, kappa=kappa)
        ext.compute_rotation_matrix()
        
        om_rec, ph_rec, kp_rec = angles_from_dm(ext.dm)
        
        ext_rec = Exterior(x0=0.0, y0=0.0, z0=0.0, omega=om_rec, phi=ph_rec, kappa=kp_rec)
        ext_rec.compute_rotation_matrix()
        
        assert np.allclose(ext.dm, ext_rec.dm, atol=1e-12)


@pytest.mark.unit
def test_exterior_from_rotation():
    """Verify exterior_from_rotation correctly creates Exterior."""
    C = np.array([120.0, -300.0, 500.0])
    omega, phi, kappa = 0.2, -0.3, 0.4
    ext_orig = Exterior(x0=C[0], y0=C[1], z0=C[2], omega=omega, phi=phi, kappa=kappa)
    ext_orig.compute_rotation_matrix()
    
    ext = exterior_from_rotation(C, ext_orig.dm)
    assert np.allclose([ext.x0, ext.y0, ext.z0], C, atol=1e-12)
    assert np.allclose(ext.dm, ext_orig.dm, atol=1e-12)


@pytest.mark.unit
def test_dm_from_lookat():
    """Verify dm_from_lookat produces viewing direction toward target."""
    C = np.array([0.0, 0.0, 1000.0])
    target = np.array([0.0, 0.0, 0.0])
    dm = dm_from_lookat(C, target, up=(0.0, 1.0, 0.0))
    
    # Camera looking along -Z in world
    ext = exterior_from_rotation(C, dm)
    assert np.allclose([ext.x0, ext.y0, ext.z0], C)


@pytest.mark.unit
def test_seed_from_lookat():
    """Verify seed_from_lookat returns valid Calibration."""
    cal = seed_from_lookat(
        position=np.array([100.0, 200.0, 800.0]),
        target=np.array([0.0, 0.0, 0.0]),
        focal_mm=35.0,
        up=(0.0, 1.0, 0.0),
    )
    assert isinstance(cal, Calibration)
    assert cal.int_par.cc == pytest.approx(35.0)
    assert cal.ext_par.z0 == pytest.approx(800.0)


@pytest.mark.unit
def test_seed_from_dlt():
    """Verify seed_from_dlt resects synthetic 3D points."""
    x = np.linspace(-100, 100, 5)
    y = np.linspace(-100, 100, 5)
    z = np.array([0.0, 50.0])
    grid = np.array([[xi, yi, zi] for xi in x for yi in y for zi in z], dtype=float)
    
    C_true = np.array([0.0, 0.0, 500.0])
    focal = 25.0
    pix = 0.01
    
    proj = []
    cpar = ControlPar(num_cams=1, imx=1000, imy=1000, pix_x=pix, pix_y=pix,
                      mm=MmNp(n1=1, n2=[1], d=[0], n3=1), chfield=0, tiff_flag=1, hp_flag=1, allCam_flag=0,
                      img_base_name=[""], cal_img_base_name=[""])
    
    for pt in grid:
        rel = pt - C_true
        px = 500.0 + (focal / pix) * (rel[0] / -rel[2])
        py = 500.0 - (focal / pix) * (rel[1] / -rel[2])
        proj.append([px, py])
        
    img_pts = np.array(proj, dtype=float)
    cal = seed_from_dlt(grid, img_pts, cpar)
    assert isinstance(cal, Calibration)
    assert cal.ext_par.z0 == pytest.approx(500.0, rel=0.1)
