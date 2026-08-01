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


def _write_rt_is(path, rows):
    """Write an rt_is file: header count + 'id x y z p0 p1 p2 p3' rows."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(rows)}\n")
        for i, (x, y, z) in enumerate(rows, start=1):
            f.write(
                "%4d %9.3f %9.3f %9.3f %4d %4d %4d %4d\n" % (i, x, y, z, 0, 1, 2, 3)
            )


def test_build_3d_figure_point_count():
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    fig = build_3d_figure(pts, frame=1000001)
    ax = fig.axes[0]
    # one scatter collection with the expected number of offsets
    assert len(ax.collections) == 1
    assert ax.collections[0].get_offsets().shape[0] == 3
    assert "1000001" in ax.get_title()
    assert "3 particles" in ax.get_title()


def test_build_3d_figure_empty():
    fig = build_3d_figure(np.empty((0, 3)), frame=42)
    ax = fig.axes[0]
    assert len(ax.collections) == 0
    assert "0 particles" in ax.get_title()


def test_create_panel_reads_rt_is(tmp_path):
    rt = tmp_path / "rt_is.1000001"
    _write_rt_is(rt, [(0.1, 0.2, 0.3), (1.1, 1.2, 1.3)])
    panel = create_3d_positions_panel(rt, frame=1000001)
    ax = panel.figure.axes[0]
    assert ax.collections[0].get_offsets().shape[0] == 2


def test_missing_file_raises(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError, OSError)):
        create_3d_positions_panel(tmp_path / "does_not_exist", frame=1)


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
