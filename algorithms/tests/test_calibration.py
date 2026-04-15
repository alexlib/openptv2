import numpy as np
import pytest
from algorithms.calibration import Exterior

EPS = 1e-6

def compare_matrix(m1, m2, eps=EPS):
    return np.allclose(m1, m2, atol=eps)

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
