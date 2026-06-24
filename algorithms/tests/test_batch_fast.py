"""Tests for batch Compiled-accelerated functions.

Verifies that batch Compiled functions produce identical results to scalar Python
versions, and measures speedup.
"""

import time

import numpy as np
import pytest

from algorithms.calibration import Calibration
from algorithms.parameters import ControlPar, VolumePar, MmNp
from algorithms.imgcoord import (
    img_coord, img_coord_batch,
    flat_image_coord, flat_image_coord_batch,
)
from algorithms.ray_tracing import ray_tracing, ray_tracing_batch
from algorithms.orientation import point_position, point_position_batch
from algorithms.trafo import (
    pixel_to_metric, pixel_to_metric_batch,
    metric_to_pixel, metric_to_pixel_batch,
)


@pytest.fixture
def setup():
    """Load calibration and parameters from test data."""
    ori_tmpl = "test_data/calibration/sym_cam{}.tif.ori"
    add_file = "test_data/calibration/cam1.tif.addpar"
    cpar = ControlPar.from_file("test_data/corresp/control.par")
    cpar.mm.n1 = 1.0
    cpar.mm.n2[:] = 1.49
    cpar.mm.n3 = 1.33
    cpar.mm.d[:] = 5.0

    cals = []
    for i in range(1, 5):
        cal = Calibration.from_file(ori_tmpl.format(i), add_file)
        cals.append(cal)

    vpar = VolumePar.from_file("test_data/corresp/criteria.par")
    return cals, cpar, vpar


def test_img_coord_batch_accuracy(setup):
    """Batch img_coord matches scalar version."""
    cals, cpar, _ = setup
    cal = cals[0]
    rng = np.random.RandomState(42)
    positions = rng.uniform(-20, 20, (100, 3))

    batch_result = img_coord_batch(positions, cal, cpar.mm)

    for i in range(len(positions)):
        x, y = img_coord(positions[i], cal, cpar.mm)
        np.testing.assert_allclose(
            batch_result[i], [x, y], atol=1e-10,
            err_msg=f"Mismatch at point {i}",
        )


def test_flat_image_coord_batch_accuracy(setup):
    """Batch flat_image_coord matches scalar version."""
    cals, cpar, _ = setup
    cal = cals[0]
    e = cal.ext_par
    ip = cal.int_par
    g = cal.glass_par
    mm = cpar.mm
    rng = np.random.RandomState(42)
    positions = rng.uniform(-20, 20, (100, 3))

    batch_result = flat_image_coord_batch(positions, cal, mm)

    for i in range(len(positions)):
        x, y = flat_image_coord(
            positions[i],
            e.x0, e.y0, e.z0, e.dm, ip.cc,
            g.vec_x, g.vec_y, g.vec_z,
            mm.n1, mm.n2[0], mm.n3, mm.d[0],
        )
        np.testing.assert_allclose(
            batch_result[i], [x, y], atol=1e-10,
            err_msg=f"Mismatch at point {i}",
        )


def test_ray_tracing_batch_accuracy(setup):
    """Batch ray_tracing matches scalar version."""
    cals, cpar, _ = setup
    cal = cals[0]
    mm = cpar.mm
    rng = np.random.RandomState(42)
    xy = rng.uniform(-5, 5, (100, 2))

    pos_batch, dir_batch = ray_tracing_batch(xy, cal, mm)

    for i in range(len(xy)):
        pos, d = ray_tracing(
            xy[i, 0], xy[i, 1],
            cal.ext_par.dm,
            cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0,
            cal.int_par.cc,
            cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z,
            mm.n1, mm.n2[0], mm.n3, mm.d[0],
        )
        np.testing.assert_allclose(
            pos_batch[i], pos, atol=1e-10,
            err_msg=f"Position mismatch at point {i}",
        )
        np.testing.assert_allclose(
            dir_batch[i], d, atol=1e-10,
            err_msg=f"Direction mismatch at point {i}",
        )


def test_point_position_batch_accuracy(setup):
    """Batch point_position matches scalar version."""
    cals, cpar, _ = setup
    mm = cpar.mm
    num_cams = 4
    rng = np.random.RandomState(42)
    positions_3d = rng.uniform(-20, 20, (50, 3))

    targets = np.empty((50, num_cams, 2), dtype=np.float64)
    for i, pos in enumerate(positions_3d):
        for cam in range(num_cams):
            x, y = flat_image_coord(
                pos,
                cals[cam].ext_par.x0, cals[cam].ext_par.y0, cals[cam].ext_par.z0,
                cals[cam].ext_par.dm, cals[cam].int_par.cc,
                cals[cam].glass_par.vec_x, cals[cam].glass_par.vec_y,
                cals[cam].glass_par.vec_z,
                mm.n1, mm.n2[0], mm.n3, mm.d[0],
            )
            targets[i, cam] = [x, y]

    pos_batch, dist_batch = point_position_batch(targets, num_cams, mm, cals)

    for i in range(len(targets)):
        pos, dist = point_position(targets[i], num_cams, mm, cals)
        np.testing.assert_allclose(
            pos_batch[i], pos, atol=1e-8,
            err_msg=f"Position mismatch at target {i}",
        )
        np.testing.assert_allclose(
            dist_batch[i], dist, atol=1e-8,
            err_msg=f"Distance mismatch at target {i}",
        )


def test_pixel_to_metric_batch_accuracy(setup):
    """Batch pixel_to_metric matches scalar version."""
    _, cpar, _ = setup
    rng = np.random.RandomState(42)
    xy = rng.uniform(0, max(cpar.imx, cpar.imy), (200, 2))

    batch_result = pixel_to_metric_batch(xy, cpar)

    for i in range(len(xy)):
        x, y = pixel_to_metric(xy[i, 0], xy[i, 1], cpar)
        np.testing.assert_allclose(
            batch_result[i], [x, y], atol=1e-12,
            err_msg=f"Mismatch at point {i}",
        )


def test_metric_to_pixel_batch_accuracy(setup):
    """Batch metric_to_pixel matches scalar version."""
    _, cpar, _ = setup
    rng = np.random.RandomState(42)
    xy = rng.uniform(-10, 10, (200, 2))

    batch_result = metric_to_pixel_batch(xy, cpar)

    for i in range(len(xy)):
        x, y = metric_to_pixel(xy[i, 0], xy[i, 1], cpar)
        np.testing.assert_allclose(
            batch_result[i], [x, y], atol=1e-12,
            err_msg=f"Mismatch at point {i}",
        )


@pytest.mark.slow
def test_batch_speedup(setup):
    """Benchmark batch vs scalar functions."""
    cals, cpar, _ = setup
    cal = cals[0]
    mm = cpar.mm
    num_cams = 4
    N = 1000

    rng = np.random.RandomState(42)
    positions = rng.uniform(-20, 20, (N, 3))
    xy_metric = rng.uniform(-5, 5, (N, 2))
    xy_pixel = rng.uniform(0, max(cpar.imx, cpar.imy), (N, 2))

    targets = np.empty((N, num_cams, 2), dtype=np.float64)
    for i, pos in enumerate(positions):
        for cam in range(num_cams):
            x, y = flat_image_coord(
                pos,
                cals[cam].ext_par.x0, cals[cam].ext_par.y0, cals[cam].ext_par.z0,
                cals[cam].ext_par.dm, cals[cam].int_par.cc,
                cals[cam].glass_par.vec_x, cals[cam].glass_par.vec_y,
                cals[cam].glass_par.vec_z,
                mm.n1, mm.n2[0], mm.n3, mm.d[0],
            )
            targets[i, cam] = [x, y]

    # Warmup Compiled
    img_coord_batch(positions[:2], cal, mm)
    flat_image_coord_batch(positions[:2], cal, mm)
    ray_tracing_batch(xy_metric[:2], cal, mm)
    point_position_batch(targets[:2], num_cams, mm, cals)
    pixel_to_metric_batch(xy_pixel[:2], cpar)
    metric_to_pixel_batch(xy_metric[:2], cpar)

    results = {}

    def bench(name, scalar_fn, batch_fn):
        t0 = time.perf_counter()
        for _ in range(3):
            scalar_fn()
        t_scalar = (time.perf_counter() - t0) / 3

        t0 = time.perf_counter()
        for _ in range(3):
            batch_fn()
        t_batch = (time.perf_counter() - t0) / 3

        ratio = t_scalar / t_batch if t_batch > 0 else float('inf')
        results[name] = (t_scalar * 1000, t_batch * 1000, ratio)

    bench("img_coord",
          lambda: [img_coord(positions[i], cal, mm) for i in range(N)],
          lambda: img_coord_batch(positions, cal, mm))

    bench("flat_image_coord",
          lambda: [flat_image_coord(
              positions[i],
              cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0,
              cal.ext_par.dm, cal.int_par.cc,
              cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z,
              mm.n1, mm.n2[0], mm.n3, mm.d[0],
          ) for i in range(N)],
          lambda: flat_image_coord_batch(positions, cal, mm))

    bench("ray_tracing",
          lambda: [ray_tracing(
              xy_metric[i, 0], xy_metric[i, 1],
              cal.ext_par.dm,
              cal.ext_par.x0, cal.ext_par.y0, cal.ext_par.z0,
              cal.int_par.cc,
              cal.glass_par.vec_x, cal.glass_par.vec_y, cal.glass_par.vec_z,
              mm.n1, mm.n2[0], mm.n3, mm.d[0],
          ) for i in range(N)],
          lambda: ray_tracing_batch(xy_metric, cal, mm))

    M = 50
    bench("point_position",
          lambda: [point_position(targets[i], num_cams, mm, cals) for i in range(M)],
          lambda: point_position_batch(targets[:M], num_cams, mm, cals))

    bench("pixel_to_metric",
          lambda: [pixel_to_metric(xy_pixel[i, 0], xy_pixel[i, 1], cpar) for i in range(N)],
          lambda: pixel_to_metric_batch(xy_pixel, cpar))

    bench("metric_to_pixel",
          lambda: [metric_to_pixel(xy_metric[i, 0], xy_metric[i, 1], cpar) for i in range(N)],
          lambda: metric_to_pixel_batch(xy_metric, cpar))

    print("\n=== Batch Compiled Speedup Results ===")
    print(f"{'Function':<25} {'Scalar (ms)':>12} {'Batch (ms)':>12} {'Speedup':>10}")
    print("-" * 65)
    for name, (t_s, t_b, ratio) in results.items():
        print(f"{name:<25} {t_s:>12.3f} {t_b:>12.3f} {ratio:>9.1f}x")
