"""Interactive 3D visualization of particle trajectories using flowtracks.

Opens a TraitsUI window hosting an embedded, mouse-rotatable matplotlib
3D plot of particle trajectories read from ptv_is files via flowtracks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from traits.api import HasTraits, Instance
from traitsui.api import Item, View

from .plot_3d_positions import MPLFigureEditor, _draw_fov_box


def _extract_xyz_mm(traj) -> np.ndarray:
    """Extract (N, 3) metric xyz position array in mm from a trajectory object or array."""
    if hasattr(traj, "pos"):
        p = traj.pos()
        if callable(p):
            p = p()
        arr = np.asarray(p, dtype=float).reshape(-1, 3)
        # flowtracks Trajectory objects store positions in meters -> convert to mm
        if arr.size > 0 and np.max(np.abs(arr)) < 20.0:
            arr = arr * 1000.0
        return arr
    else:
        # Raw position array input is in mm (OpenPTV2 standard)
        arr = np.asarray(traj, dtype=float).reshape(-1, 3)
        return arr


def build_3d_trajectories_figure(
    trajectories: Sequence,
    bounds=None,
    first_frame: int | None = None,
    last_frame: int | None = None,
    total_frames_requested: int | None = None,
    max_step: float = 50.0,
) -> Figure:
    """Build a 3D line plot Figure for particle trajectories.

    Pure function (no window): safe to call under the Agg backend in tests.

    Args:
        trajectories: Sequence of flowtracks Trajectory objects or (N, 3) position arrays.
        bounds: optional ((xmin, xmax), (ymin, ymax), (zmin, zmax)) axis limits.
        first_frame: first frame number displayed.
        last_frame: last frame number displayed.
        total_frames_requested: total frames requested by user before clamping to 50.

    Returns:
        A matplotlib Figure containing a 3D axes with plotted trajectories.
    """
    fig = Figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    num_trajs = len(trajectories) if trajectories is not None else 0

    try:
        cmap = matplotlib.colormaps.get_cmap("tab20")
    except AttributeError:
        cmap = plt.cm.get_cmap("tab20")

    valid_trajs = []
    all_pos = []
    for traj in trajectories:
        tid = getattr(traj, "trajid", lambda: 1)() if hasattr(traj, "trajid") else 1
        if callable(tid):
            try:
                tid = tid()
            except Exception:
                tid = 1
        if tid > 0:
            valid_trajs.append(traj)

    if len(valid_trajs) > 0:
        for idx, traj in enumerate(valid_trajs):
            pos = _extract_xyz_mm(traj)
            if pos.shape[0] > 0:
                all_pos.append(pos)
                t = None
                if hasattr(traj, "time"):
                    try:
                        t = traj.time()
                        if callable(t):
                            t = t()
                        t = np.asarray(t)
                        if len(t) == pos.shape[0]:
                            order = np.argsort(t)
                            pos = pos[order]
                            t = t[order]
                    except Exception:
                        t = None

                # Break into continuous physical segments (dt == 1, step <= 10.0 mm)
                segments = []
                if pos.shape[0] == 1 or t is None or len(t) != pos.shape[0]:
                    segments = [pos]
                else:
                    curr_seg = [pos[0]]
                    for i in range(len(t) - 1):
                        dt = t[i + 1] - t[i]
                        dp = np.linalg.norm(pos[i + 1] - pos[i])
                        if dt == 1 and dp <= max_step:
                            curr_seg.append(pos[i + 1])
                        else:
                            if len(curr_seg) > 0:
                                segments.append(np.array(curr_seg))
                            curr_seg = [pos[i + 1]]
                    if len(curr_seg) > 0:
                        segments.append(np.array(curr_seg))

                color = cmap(idx % 20)
                for seg in segments:
                    if seg.shape[0] > 0:
                        # Default view (see ax.view_init below) puts mpl's
                        # y-axis on our z (depth, near-camera = positive) and
                        # mpl's z-axis on our y (vertical) -- feed the plot
                        # accordingly: (x, z, y) not (x, y, z).
                        ax.plot(
                            seg[:, 0],
                            seg[:, 2],
                            seg[:, 1],
                            linewidth=1.5,
                            color=color,
                            alpha=0.8,
                        )

    if bounds is not None:
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = bounds
    elif all_pos:
        pts = np.concatenate(all_pos, axis=0)
        xlo, xhi = float(pts[:, 0].min()), float(pts[:, 0].max())
        ylo, yhi = float(pts[:, 1].min()), float(pts[:, 1].max())
        zlo, zhi = float(pts[:, 2].min()), float(pts[:, 2].max())
    else:
        xlo = xhi = ylo = yhi = zlo = zhi = None

    if xlo is not None:
        if bounds is not None:
            _draw_fov_box(ax, xlo, xhi, zlo, zhi, ylo, yhi)
            ax.legend(loc="upper left", fontsize=8)
        mx = (xhi - xlo) * 0.05 or 1.0
        my = (yhi - ylo) * 0.05 or 1.0
        mz = (zhi - zlo) * 0.05 or 1.0
        ax.set_xlim(xlo - mx, xhi + mx)
        ax.set_ylim(zlo - mz, zhi + mz)
        ax.set_zlim(ylo - my, yhi + my)

    # Equal physical aspect: without this mpl stretches each axis to fill the
    # plot cube independently, which visually exaggerates z-motion in shallow
    # volumes and makes correct trajectory links look like erratic zigzags.
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    zlo, zhi = ax.get_zlim()
    ax.set_box_aspect((xhi - xlo or 1.0, yhi - ylo or 1.0, zhi - zlo or 1.0))

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("z (mm)")
    ax.set_zlabel("y (mm)")

    # Default view: x left-right, y bottom-up, z close to camera = positive.
    ax.view_init(elev=20, azim=100)
    ax.invert_xaxis()

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

    ax.set_title(title)
    return fig


class Plot3DTrajectories(HasTraits):
    """TraitsUI window hosting the embedded 3D matplotlib trajectories plot."""

    figure = Instance(Figure)

    traits_view = View(
        Item("figure", editor=MPLFigureEditor(), show_label=False),
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
    """Read ptv_is files or Zarr store using flowtracks and return Plot3DTrajectories window.

    If total frames (last_frame - first_frame + 1) > 50, loads only the first 50 frames.
    """
    total_frames = last_frame - first_frame + 1
    effective_last = last_frame
    if total_frames > 50:
        effective_last = first_frame + 49

    trajectories = []
    from openptv2.storage import find_existing_store

    zarr_store = find_existing_store(exp_path)
    if zarr_store is not None:
        try:
            from openptv2.storage import RunStore

            store = RunStore(zarr_store, mode="a")
            trajectories = store.to_flowtracks_trajectories(
                first=first_frame, last=effective_last, traj_min_len=traj_min_len
            )
        except Exception:
            trajectories = []

    if not trajectories:
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

    fig = build_3d_trajectories_figure(
        trajectories,
        bounds=bounds,
        first_frame=first_frame,
        last_frame=effective_last,
        total_frames_requested=total_frames,
    )
    return Plot3DTrajectories(figure=fig)

