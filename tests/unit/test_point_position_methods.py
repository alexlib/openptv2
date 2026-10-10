"""Triangulation methods of point_positions: "pairs" (default), "weighted" and "lsq".

The default averages the skew-line midpoints of all camera pairs. In a ring of four cameras two of the six pairs
look along nearly the same line, and their midpoints are poorly determined along it, so the average amplifies image
noise. "weighted" (sin^2 pair-angle weights) and "lsq" (least squares over all rays) remove that.
"""
import numpy as np
import pytest

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.imgcoord import flat_image_coord_batch
from openptv2.algorithms.orientation import (
    point_position_batch,
    point_positions,
)
from openptv2.algorithms.parameters import ControlPar, MmNp

PIX = 5.5e-3
CC = 50.0


def _angles_from_matrix(m):
    """Angles (omega, phi, kappa) such that M = Rx(omega) Ry(phi) Rz(kappa)."""
    phi = np.arcsin(m[0, 2])
    return np.array([np.arctan2(-m[1, 2], m[2, 2]), phi, np.arctan2(-m[0, 1], m[0, 0])])


def _ring(azimuths_deg, distance=450.0, glass=150.0):
    """Cameras on a horizontal ring looking at the origin, flat glass window per camera."""
    mm = MmNp(n1=1.0, n2=[1.5], d=[5.0], n3=1.333, nlay=1)
    cpar = ControlPar(len(azimuths_deg), imx=2048, imy=2048, pix_x=PIX, pix_y=PIX, mm=mm)
    cals = []
    for az in np.deg2rad(azimuths_deg):
        pos = np.array([distance * np.cos(az), distance * np.sin(az), 0.0])
        zc = -pos / np.linalg.norm(pos)
        xc = np.cross(zc, [0, 0, 1.0])
        xc /= np.linalg.norm(xc)
        yc = np.cross(zc, xc)
        m = np.vstack([xc, yc, zc]).T @ np.diag([1.0, -1.0, -1.0])
        cal = Calibration()
        cal.set_pos(pos)
        cal.set_angles(_angles_from_matrix(m))
        cal.set_primary_point(np.array([0.0, 0.0, CC]))
        radial = pos / np.linalg.norm(pos)
        cal.set_glass_vec(radial * (glass - 5.0))  # water-side face of the glass
        cals.append(cal)
    return cpar, cals


def _points(n, seed=0, r=40.0, h=40.0):
    g = np.random.default_rng(seed)
    rr = r * np.sqrt(g.random(n))
    th = 2 * np.pi * g.random(n)
    return np.c_[rr * np.cos(th), rr * np.sin(th), g.uniform(-h, h, n)]


def _targets(points, cpar, cals, noise_px=0.0, seed=1):
    xy = [np.asarray(flat_image_coord_batch(points, cal, cpar.mm)) for cal in cals]
    t = np.stack(xy, axis=1)
    return t + np.random.default_rng(seed).normal(0.0, noise_px * PIX, t.shape)


def _rms(a, b):
    return float(np.sqrt((np.linalg.norm(a - b, axis=1) ** 2).mean()))


RING4 = (40.0, 140.0, 220.0, 320.0)  # pairs 40/220 and 140/320 are 173 degrees apart


@pytest.mark.parametrize("method", ["pairs", "weighted", "lsq"])
def test_noise_free_ring_recovers_points(method):
    cpar, cals = _ring(RING4)
    x = _points(60)
    pos, _ = point_positions(_targets(x, cpar, cals), cpar, cals, method=method)
    assert _rms(pos, x) < 1e-3  # forward projection of the library is iterative, accurate to about 3e-5 mm


def test_default_is_pairs_and_unchanged():
    cpar, cals = _ring(RING4)
    x = _points(40)
    t = _targets(x, cpar, cals, noise_px=0.1)
    a, da = point_positions(t, cpar, cals)
    b, db = point_positions(t, cpar, cals, method="pairs")
    assert np.array_equal(a, b) and np.array_equal(da, db)


def test_two_cameras_all_methods_agree():
    cpar, cals = _ring((40.0, 140.0))
    x = _points(40)
    t = _targets(x, cpar, cals, noise_px=0.3)
    ref, _ = point_positions(t, cpar, cals, method="pairs")
    for method in ("weighted", "lsq"):
        pos, _ = point_positions(t, cpar, cals, method=method)
        assert np.allclose(pos, ref, atol=1e-9)


def test_unused_cameras_are_skipped():
    from openptv2.algorithms.constants import COORD_UNUSED

    cpar, cals = _ring(RING4)
    x = _points(20)
    t = _targets(x, cpar, cals, noise_px=0.1)
    t[:, 1, :] = COORD_UNUSED  # camera 1 does not see the points
    ref, _ = point_positions(t, cpar, cals, method="pairs")
    for method in ("weighted", "lsq"):
        pos, _ = point_positions(t, cpar, cals, method=method)
        assert np.all(np.isfinite(pos))
        assert _rms(pos, x) < 5 * _rms(ref, x) + 1e-3


def test_ray_convergence_measure_is_method_independent():
    cpar, cals = _ring(RING4)
    t = _targets(_points(30), cpar, cals, noise_px=0.2)
    _, d0 = point_positions(t, cpar, cals, method="pairs")
    for method in ("weighted", "lsq"):
        _, d = point_positions(t, cpar, cals, method=method)
        assert np.allclose(d, d0)


def test_near_opposite_pairs_amplify_noise_only_in_pair_average():
    cpar, cals = _ring(RING4)
    x = _points(300, seed=3)
    t = _targets(x, cpar, cals, noise_px=0.1, seed=3)
    err = {m: _rms(point_positions(t, cpar, cals, method=m)[0], x) for m in ("pairs", "weighted", "lsq")}
    assert err["pairs"] > 3.0 * err["lsq"]
    assert err["weighted"] < 1.5 * err["lsq"]


def test_matches_independent_least_squares():
    """"lsq" equals the closed-form point nearest to the rays, computed here from the rays themselves."""
    cpar, cals = _ring(RING4)
    t = _targets(_points(25), cpar, cals, noise_px=0.2)
    pos, _ = point_positions(t, cpar, cals, method="lsq")
    from openptv2.algorithms.ray_tracing import ray_tracing_batch

    rays = [ray_tracing_batch(t[:, k, :], cals[k], cpar.mm) for k in range(len(cals))]
    for i in range(len(pos)):
        a = np.zeros((3, 3))
        b = np.zeros(3)
        for p, d in rays:
            u = d[i] / np.linalg.norm(d[i])
            proj = np.eye(3) - np.outer(u, u)
            a += proj
            b += proj @ p[i]
        assert np.allclose(pos[i], np.linalg.solve(a, b), atol=1e-8)


def test_unknown_method_raises():
    cpar, cals = _ring(RING4)
    t = _targets(_points(3), cpar, cals)
    with pytest.raises(ValueError, match="unknown point-position method"):
        point_positions(t, cpar, cals, method="median")


def test_batch_entry_point_accepts_method():
    cpar, cals = _ring(RING4)
    x = _points(10)
    t = _targets(x, cpar, cals)
    pos, d = point_position_batch(t, len(cals), cpar.mm, cals, "lsq")
    assert pos.shape == (10, 3) and d.shape == (10,)
    assert _rms(pos, x) < 1e-3  # forward projection of the library is iterative, accurate to about 3e-5 mm
