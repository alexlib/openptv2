"""Tests for the 3D-trajectories visualization figure builder and GUI action.

Only the pure, headless figure builder, panel reader, and handler logic are tested —
the live TraitsUI/Qt window is not exercised here.
"""

from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")  # headless: no window, safe in CI

import numpy as np
from flowtracks.trajectory import Trajectory

from openptv2.gui.plot_3d_trajectories import (
    build_3d_trajectories_figure,
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


def test_build_3d_trajectories_figure_empty():
    fig = build_3d_trajectories_figure([], first_frame=1, last_frame=10)
    ax = fig.axes[0]
    assert len(ax.lines) == 0
    assert "0 trajectories" in ax.get_title()


def test_build_3d_trajectories_figure_count():
    traj1 = _make_dummy_trajectory(
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]), trajid=1
    )
    traj2 = _make_dummy_trajectory(
        np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0], [7.0, 7.0, 7.0]]), trajid=2
    )
    fig = build_3d_trajectories_figure(
        [traj1, traj2], first_frame=10000, last_frame=10005
    )
    ax = fig.axes[0]
    assert len(ax.lines) == 2
    assert "2 trajectories" in ax.get_title()
    assert "10000–10005" in ax.get_title() or "10000" in ax.get_title()


def test_build_3d_trajectories_figure_frame_clamping_title():
    traj = _make_dummy_trajectory(
        np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), trajid=1
    )
    fig = build_3d_trajectories_figure(
        [traj],
        first_frame=1,
        last_frame=50,
        total_frames_requested=100,
    )
    ax = fig.axes[0]
    assert "first 50 of 100 frames" in ax.get_title()


def test_bounds_and_fov_box():
    traj = _make_dummy_trajectory(
        np.array([[0.0, 0.0, -50.0], [10.0, 5.0, -30.0]]), trajid=1
    )
    bounds = ((-30.0, 50.0), (-40.0, 40.0), (-80.0, -15.0))
    fig = build_3d_trajectories_figure([traj], bounds=bounds)
    ax = fig.axes[0]
    # 1 trajectory line + 12 FOV box lines = 13 lines total
    assert len(ax.lines) == 13


def test_create_3d_trajectories_panel_clamping(tmp_path):
    res_dir = tmp_path / "res"
    res_dir.mkdir()

    with patch("flowtracks.io.trajectories_ptvis") as mock_ptvis:
        mock_ptvis.return_value = []
        # Requesting 100 frames (1 to 100)
        create_3d_trajectories_panel(tmp_path, first_frame=1, last_frame=100)
        # Should clamp last_frame to 1 + 49 = 50
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
        patch(
            "openptv2.gui.plot_3d_trajectories.Plot3DTrajectories.configure_traits"
        ) as mock_config,
    ):
        handler.visualize_3d_trajectories(mock_info)
        mock_config.assert_called_once()


def test_extract_xyz_mm_scaling():
    from openptv2.gui.plot_3d_trajectories import _extract_xyz_mm

    # Trajectory 1 with position < 100 mm (e.g. 50 mm)
    t1 = _make_dummy_trajectory(np.array([[10.0, 20.0, 50.0], [12.0, 22.0, 52.0]]))
    # Trajectory 2 with position > 100 mm (e.g. 150 mm)
    t2 = _make_dummy_trajectory(np.array([[10.0, 20.0, 150.0], [12.0, 22.0, 152.0]]))

    ext1 = _extract_xyz_mm(t1)
    ext2 = _extract_xyz_mm(t2)

    # Both must be in mm (50 mm and 150 mm)
    np.testing.assert_allclose(ext1[0], [10.0, 20.0, 50.0])
    np.testing.assert_allclose(ext2[0], [10.0, 20.0, 150.0])

    # Raw mm numpy array
    raw = np.array([[10.0, 20.0, 50.0]])
    ext_raw = _extract_xyz_mm(raw)
    np.testing.assert_allclose(ext_raw[0], [10.0, 20.0, 50.0])


def test_create_3d_trajectories_panel_zarr(tmp_path):
    res_dir = tmp_path / "res"
    res_dir.mkdir()
    zarr_dir = res_dir / "run.zarr"

    traj = _make_dummy_trajectory(np.array([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]]))

    with patch("openptv2.storage.read_zarr_trajectories", return_value=[traj]) as mock_read_zarr:
        zarr_dir.mkdir()
        panel = create_3d_trajectories_panel(tmp_path, first_frame=1, last_frame=10)
        mock_read_zarr.assert_called_once()
        ax = panel.figure.axes[0]
        assert len(ax.lines) == 1

