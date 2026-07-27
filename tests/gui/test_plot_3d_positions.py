"""Tests for the 3D-positions visualization PyVista plotter builder.

Only the pure, headless plotter builder and rt_is reader are tested — the live
TraitsUI/Qt window is not exercised here.
"""

import numpy as np
import pytest
import pyvista as pv

pv.OFF_SCREEN = True

from openptv2.gui.plot_3d_positions import (
    build_3d_positions_plotter,
    compute_fov_bounds,
    create_3d_positions_panel,
)


def _write_rt_is(path, rows):
    """Write an rt_is file: header count + 'id x y z p0 p1 p2 p3' rows."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(rows)}\n")
        for i, (x, y, z) in enumerate(rows, start=1):
            f.write("%4d %9.3f %9.3f %9.3f %4d %4d %4d %4d\n" % (i, x, y, z, 0, 1, 2, 3))


def test_build_3d_positions_plotter_point_count():
    pts = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    plotter = build_3d_positions_plotter(pts, frame=1000001)
    # Check that mesh / actors were added
    assert len(plotter.renderer.actors) >= 1


def test_build_3d_positions_plotter_empty():
    plotter = build_3d_positions_plotter(np.empty((0, 3)), frame=42)
    assert plotter is not None


def test_create_panel_reads_rt_is(tmp_path):
    rt = tmp_path / "rt_is.1000001"
    _write_rt_is(rt, [(0.1, 0.2, 0.3), (1.1, 1.2, 1.3)])
    panel = create_3d_positions_panel(rt, frame=1000001)
    assert len(panel.points) == 2


def test_missing_file_raises(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError, OSError)):
        create_3d_positions_panel(tmp_path / "does_not_exist", frame=1)


def test_bounds_and_fov_box():
    pts = np.array([[0.0, 0.0, -50.0], [10.0, 5.0, -30.0]])
    bounds = ((-30.0, 50.0), (-40.0, 40.0), (-80.0, -15.0))
    plotter = build_3d_positions_plotter(pts, frame=1, bounds=bounds)
    assert len(plotter.renderer.actors) >= 2


def test_compute_fov_bounds_from_vpar():
    from openptv2.algorithms.parameters import VolumePar

    vpar = VolumePar(X_lay=[-30, 50], Zmin_lay=[-80, -80], Zmax_lay=[-15, -15])
    xlim, ylim, zlim = compute_fov_bounds(vpar)
    assert xlim == (-30.0, 50.0)
    assert zlim == (-80.0, -15.0)
    assert ylim == xlim
