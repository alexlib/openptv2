"""Interactive 3D visualization of particle trajectories using PyVista.

NOTE: This branch fails with PyVista on Windows due to PySide6 / pyvistaqt
QVTKRenderWindowInteractor integration issues with TraitsUI windows.

Opens a TraitsUI window hosting an embedded, mouse-rotatable PyVista
3D visualization of particle trajectories read from ptv_is files via flowtracks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from traits.api import HasTraits, Instance
from traitsui.api import Item, View

from .plot_3d_positions import PyVistaEditor, compute_fov_bounds


def _extract_xyz_mm(traj) -> np.ndarray:
    """Extract (N, 3) metric xyz position array in mm from a trajectory object or array."""
    if hasattr(traj, "pos"):
        p = traj.pos()
        if callable(p):
            p = p()
        arr = np.asarray(p, dtype=float).reshape(-1, 3)
        if arr.size > 0 and np.max(np.abs(arr)) < 100.0:
            arr = arr * 1000.0
        return arr
    else:
        arr = np.asarray(traj, dtype=float).reshape(-1, 3)
        return arr


def build_3d_trajectories_plotter(
    trajectories: Sequence,
    bounds=None,
    first_frame: int | None = None,
    last_frame: int | None = None,
    total_frames_requested: int | None = None,
    plotter: pv.Plotter | None = None,
) -> pv.Plotter:
    """Build a PyVista Plotter displaying 3D particle trajectories.

    Pure function: safe to call off-screen in tests or headless environments.

    Args:
        trajectories: Sequence of flowtracks Trajectory objects or (N, 3) position arrays.
        bounds: optional ((xmin, xmax), (ymin, ymax), (zmin, zmax)) axis limits.
        first_frame: first frame number displayed.
        last_frame: last frame number displayed.
        total_frames_requested: total frames requested by user before clamping to 50.
        plotter: optional existing PyVista Plotter / QtInteractor instance.

    Returns:
        The populated PyVista Plotter instance.
    """
    if plotter is None:
        plotter = pv.Plotter(off_screen=True)

    num_trajs = len(trajectories) if trajectories is not None else 0

    try:
        cmap = matplotlib.colormaps.get_cmap("tab20")
    except AttributeError:
        cmap = plt.cm.get_cmap("tab20")

    if num_trajs > 0:
        for idx, traj in enumerate(trajectories):
            pos = _extract_xyz_mm(traj)
            n_pts = pos.shape[0]
            if n_pts >= 2:
                rgb = cmap(idx % 20)[:3]
                line = pv.lines_from_points(pos)
                plotter.add_mesh(line, color=rgb, line_width=3)
                plotter.add_mesh(
                    pv.PolyData(pos),
                    color=rgb,
                    point_size=6,
                    render_points_as_spheres=True,
                )
            elif n_pts == 1:
                rgb = cmap(idx % 20)[:3]
                plotter.add_mesh(
                    pv.PolyData(pos),
                    color=rgb,
                    point_size=8,
                    render_points_as_spheres=True,
                )

    if bounds is not None:
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = bounds
        box = pv.Box(bounds=[xlo, xhi, ylo, yhi, zlo, zhi])
        plotter.add_mesh(
            box,
            style="wireframe",
            color="#d1495b",
            line_width=2.0,
            label="field of view",
        )

    plotter.add_axes(xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)")
    plotter.show_grid(xtitle="x (mm)", ytitle="y (mm)", ztitle="z (mm)")

    if first_frame is not None and last_frame is not None:
        if total_frames_requested is not None and total_frames_requested > 50:
            title = (
                f"3D trajectories — frames {first_frame}–{last_frame} "
                f"(first 50 of {total_frames_requested} frames, {num_trajs} trajectories)"
            )
        else:
            title = (
                f"3D trajectories — frames {first_frame}–{last_frame} "
                f"({num_trajs} trajectories)"
            )
    else:
        title = f"3D trajectories ({num_trajs} trajectories)"

    plotter.add_title(title, font_size=12)
    plotter.reset_camera()
    return plotter


class Plot3DTrajectories(HasTraits):
    """TraitsUI window hosting the embedded 3D PyVista plot for trajectories."""

    plotter_builder = Instance(object)

    traits_view = View(
        Item("plotter_builder", editor=PyVistaEditor(), show_label=False),
        title="Visualize 3D trajectories",
        resizable=True,
        width=900,
        height=700,
    )


def create_3d_trajectories_panel(
    exp_path: str | Path,
    first_frame: int,
    last_frame: int,
    bounds=None,
    xuap: bool = False,
    traj_min_len: int = 2,
) -> Plot3DTrajectories:
    """Read ptv_is files using flowtracks and return Plot3DTrajectories window.

    If total frames (last_frame - first_frame + 1) > 50, loads only the first 50 frames.
    """
    total_frames = last_frame - first_frame + 1
    effective_last = last_frame
    if total_frames > 50:
        effective_last = first_frame + 49

    ptv_is_pattern = str(Path(exp_path) / "res" / "ptv_is.%d")

    try:
        from flowtracks.io import trajectories_ptvis

        trajectories = trajectories_ptvis(
            ptv_is_pattern,
            first=first_frame,
            last=effective_last,
            xuap=xuap,
            traj_min_len=traj_min_len,
        )
    except Exception:
        trajectories = []

    def builder(interactor):
        build_3d_trajectories_plotter(
            trajectories,
            bounds=bounds,
            first_frame=first_frame,
            last_frame=effective_last,
            total_frames_requested=total_frames,
            plotter=interactor,
        )

    panel = Plot3DTrajectories(plotter_builder=builder)
    panel.trajectories = trajectories
    return panel
