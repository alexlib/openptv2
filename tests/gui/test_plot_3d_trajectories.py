"""Tests for the 3D-trajectories visualization PyVista plotter builder and GUI action.

Only the pure, headless plotter builder, panel reader, and handler logic are tested —
the live TraitsUI/Qt window is not exercised here.
"""

from unittest.mock import MagicMock, patch
import pyvista as pv

pv.OFF_SCREEN = True

from flowtracks.trajectory import Trajectory
import numpy as np
import pytest

from openptv2.gui.plot_3d_trajectories import (
    build_3d_trajectories_plotter,
    create_3d_trajectories_panel,
)
from openptv2.gui.pyptv_gui import TreeMenuHandler


def _make_dummy_trajectory(pos_mm: np.ndarray, trajid: int = 1) -> Trajectory:
    """Helper to build a flowtracks Trajectory instance from (N, 3) mm coords."""
    n = pos_mm.shape[0]
    pos_m = pos_mm / 1000.0  # flowtracks stores meters
    vel = np.zeros_like(pos_m)
    time = np.arange(10000, 10000 + n, dtype=float)
    return Trajectory(pos_m, vel, time, trajid)


def test_build_3d_trajectories_plotter_empty():
    plotter = build_3d_trajectories_plotter([], first_frame=1, last_frame=10)
    assert plotter is not None


def test_build_3d_trajectories_plotter_count():
    traj1 = _make_dummy_trajectory(
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]), trajid=1
    )
    traj2 = _make_dummy_trajectory(
        np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0], [7.0, 7.0, 7.0]]), trajid=2
    )
    plotter = build_3d_trajectories_plotter([traj1, traj2], first_frame=10000, last_frame=10005)
    assert len(plotter.renderer.actors) >= 2


def test_build_3d_trajectories_plotter_frame_clamping_title():
    traj = _make_dummy_trajectory(
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), trajid=1
    )
    plotter = build_3d_trajectories_plotter(
        [traj],
        first_frame=1,
        last_frame=50,
        total_frames_requested=100,
    )
    assert plotter is not None


def test_bounds_and_fov_box():
    traj = _make_dummy_trajectory(
        np.array([[0.0, 0.0, -50.0], [10.0, 5.0, -30.0]]), trajid=1
    )
    bounds = ((-30.0, 50.0), (-40.0, 40.0), (-80.0, -15.0))
    plotter = build_3d_trajectories_plotter([traj], bounds=bounds)
    assert len(plotter.renderer.actors) >= 2


def test_create_3d_trajectories_panel_clamping(tmp_path):
    res_dir = tmp_path / "res"
    res_dir.mkdir()

    with patch("flowtracks.io.trajectories_ptvis") as mock_ptvis:
        mock_ptvis.return_value = []
        panel = create_3d_trajectories_panel(tmp_path, first_frame=1, last_frame=100)
        mock_ptvis.assert_called_once()
        _, kwargs = mock_ptvis.call_args
        assert kwargs["first"] == 1
        assert kwargs["last"] == 50


def test_visualize_3d_trajectories_no_files(tmp_path):
    handler = TreeMenuHandler()
    mock_info = MagicMock()
    mock_gui = MagicMock()
    mock_gui.exp_path = str(tmp_path)
    mock_gui.get_parameter.return_value = {"first": 1, "last": 10}
    mock_info.object = mock_gui

    with patch("pyface.api.warning") as mock_warning:
        handler.visualize_3d_trajectories(mock_info)
        mock_warning.assert_called_once()


def test_visualize_3d_trajectories_success(tmp_path):
    res_dir = tmp_path / "res"
    res_dir.mkdir()
    (res_dir / "ptv_is.10000").touch()

    handler = TreeMenuHandler()
    mock_info = MagicMock()
    mock_gui = MagicMock()
    mock_gui.exp_path = str(tmp_path)
    mock_gui.get_parameter.return_value = {"first": 10000, "last": 10010}
    mock_info.object = mock_gui

    traj = _make_dummy_trajectory(
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), trajid=1
    )

    with (
        patch("flowtracks.io.trajectories_ptvis", return_value=[traj]),
        patch("openptv2.gui.plot_3d_trajectories.Plot3DTrajectories.configure_traits") as mock_config,
    ):
        handler.visualize_3d_trajectories(mock_info)
        mock_config.assert_called_once()
