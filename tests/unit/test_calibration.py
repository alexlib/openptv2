import os

import numpy as np

from openptv2.algorithms.calibration import (
    AddedPar,
    Calibration,
    Exterior,
    Glass,
    Interior,
)

EPS = 1e-6

def compare_matrix(m1, m2, eps=EPS):
    return np.allclose(m1, m2, atol=eps)

def make_test_cal():
    # Helper to generate test_cal like in C
    ext = Exterior(
        x0=105.2632, y0=102.7458, z0=403.8822,
        omega=-0.2383291, phi=0.2442810, kappa=0.0552577,
        dm=np.array([
            [0.9688305, -0.0535899, 0.2418587],
            [-0.0033422, 0.9734041, 0.2290704],
            [-0.2477021, -0.2227387, 0.9428845]
        ])
    )
    int_par = Interior(xh=-2.4742, yh=3.2567, cc=100.0000)
    glass = Glass(vec_x=0.0001, vec_y=0.00001, vec_z=150.0)
    addp = AddedPar(k1=0., k2=0., k3=0., p1=0., p2=0., scx=1., she=0.)

    cal = Calibration(ext_par=ext, int_par=int_par, glass_par=glass, added_par=addp)
    cal.ext_par.compute_rotation_matrix()
    return cal

def test_read_ori():
    correct_cal = make_test_cal()
    ori_file = "test_data/calibration/cam1.tif.ori"
    add_file = "test_data/calibration/cam1.tif.addpar"

    cal = Calibration.from_file(ori_file, add_file)

    assert np.allclose(cal.ext_par.x0, correct_cal.ext_par.x0)
    assert np.allclose(cal.ext_par.y0, correct_cal.ext_par.y0)
    assert np.allclose(cal.ext_par.z0, correct_cal.ext_par.z0)
    assert np.allclose(cal.ext_par.omega, correct_cal.ext_par.omega)
    assert np.allclose(cal.ext_par.phi, correct_cal.ext_par.phi)
    assert np.allclose(cal.ext_par.kappa, correct_cal.ext_par.kappa)
    assert np.allclose(cal.ext_par.dm, correct_cal.ext_par.dm, atol=1e-6)

    assert np.allclose(cal.int_par.xh, correct_cal.int_par.xh)
    assert np.allclose(cal.int_par.yh, correct_cal.int_par.yh)
    assert np.allclose(cal.int_par.cc, correct_cal.int_par.cc)

    assert np.allclose(cal.glass_par.vec_x, correct_cal.glass_par.vec_x)
    assert np.allclose(cal.glass_par.vec_y, correct_cal.glass_par.vec_y)
    assert np.allclose(cal.glass_par.vec_z, correct_cal.glass_par.vec_z)

    assert np.allclose(cal.added_par.k1, correct_cal.added_par.k1)
    assert np.allclose(cal.added_par.k2, correct_cal.added_par.k2)
    assert np.allclose(cal.added_par.k3, correct_cal.added_par.k3)
    assert np.allclose(cal.added_par.p1, correct_cal.added_par.p1)
    assert np.allclose(cal.added_par.p2, correct_cal.added_par.p2)
    assert np.allclose(cal.added_par.scx, correct_cal.added_par.scx)
    assert np.allclose(cal.added_par.she, correct_cal.added_par.she)

def test_write_ori():
    correct_cal = make_test_cal()
    ori_file = "test_data/test.ori"
    add_file = "test_data/test.addpar"

    correct_cal.to_file(ori_file, add_file)

    cal = Calibration.from_file(ori_file, add_file)

    assert np.allclose(cal.ext_par.x0, correct_cal.ext_par.x0)
    assert np.allclose(cal.ext_par.dm, correct_cal.ext_par.dm, atol=1e-6)

    os.remove(ori_file)
    os.remove(add_file)

def test_rotation_angles():
    # omega
    ex = Exterior()
    ex.omega = np.pi / 2.0
    ex.phi = 0.0
    ex.kappa = 0.0
    rotx = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    ex.compute_rotation_matrix()
    assert compare_matrix(ex.dm, rotx)

    # phi
    ex = Exterior()
    ex.omega = 0.0
    ex.phi = np.pi / 2.0
    ex.kappa = 0.0
    roty = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    ex.compute_rotation_matrix()
    assert compare_matrix(ex.dm, roty)

    # kappa
    ex = Exterior()
    ex.omega = 0.0
    ex.phi = 0.0
    ex.kappa = np.pi / 2.0
    rotz = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    ex.compute_rotation_matrix()
    assert compare_matrix(ex.dm, rotz)
