"""Tests for the 3D-positions visualization figure builder.

Only the pure, headless figure builder and rt_is reader are tested — the live
TraitsUI/Qt window is not exercised here.
"""

import matplotlib

matplotlib.use("Agg")  # headless: no window, safe in CI

import numpy as np
import pytest

from openptv2.gui.plot_3d_positions import (
    build_3d_figure,
    compute_fov_bounds,
    create_3d_positions_panel,
)


def test_create_panel_reads_store(tmp_path):
    from openptv2.storage import RunStore

    store = RunStore(tmp_path / "res" / "run.zarr", mode="w")
    pts = np.array([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]], dtype=np.float64)
    cam_ids = np.zeros((2, 4), dtype=np.int32)
    store.write_correspondences(1000001, pts, cam_ids)
    panel = create_3d_positions_panel(tmp_path, frame=1000001)
    ax = panel.figure.axes[0]
    assert ax.collections[0].get_offsets().shape[0] == 2


def test_missing_store_returns_empty_panel(tmp_path):
    panel = create_3d_positions_panel(tmp_path / "does_not_exist", frame=1)
    assert len(panel.figure.axes[0].collections) == 0


def test_bounds_and_fov_box():
    # An outlier far outside the FOV must not blow up the axes, and the
    # 12-edge FOV cuboid must be drawn.
    pts = np.array([[0.0, 0.0, -50.0], [10.0, 5.0, -30.0], [9e4, 9e4, 9e4]])
    bounds = ((-30.0, 50.0), (-40.0, 40.0), (-80.0, -15.0))
    fig = build_3d_figure(pts, frame=1, bounds=bounds)
    ax = fig.axes[0]
    # 12 box edges drawn as Line3D objects
    assert len(ax.lines) == 12
    # Axes clamped near the FOV (padded by 5%), not stretched to 9e4
    xlo, xhi = ax.get_xlim3d()
    assert -40.0 <= xlo <= -25.0 and 45.0 <= xhi <= 60.0
    zlo, zhi = ax.get_zlim3d()
    assert -90.0 <= zlo <= -75.0 and -20.0 <= zhi <= -10.0


def test_compute_fov_bounds_from_vpar():
    from openptv2.algorithms.parameters import VolumePar

    vpar = VolumePar(X_lay=[-30, 50], Zmin_lay=[-80, -80], Zmax_lay=[-15, -15])
    xlim, ylim, zlim = compute_fov_bounds(vpar)  # no cpar/cals -> Y falls back
    assert xlim == (-30.0, 50.0)
    assert zlim == (-80.0, -15.0)
    assert ylim == xlim  # square-FOV fallback when Y cannot be derived
