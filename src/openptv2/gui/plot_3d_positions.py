"""Interactive 3D visualization of rt_is particle positions.

Opens a TraitsUI window (same pattern as the other auxiliary GUI windows,
via ``configure_traits()``) hosting an embedded, mouse-rotatable matplotlib
3D scatter of the 3D positions stored in an ``rt_is.<frame>`` file.

Chaco has no 3D renderer, so the plot content is matplotlib embedded in a
TraitsUI window through a small ``MPLFigureEditor`` (the standard Enthought
recipe), keeping everything inside the TraitsUI/Qt event loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from traits.api import HasTraits, Instance
from traitsui.api import Item, View
from traitsui.basic_editor_factory import BasicEditorFactory
from traitsui.qt.editor import Editor

from . import ptv


class _MPLFigureEditor(Editor):
    """Embeds a matplotlib Figure as a Qt canvas inside a TraitsUI Item."""

    scrollable = True

    def init(self, parent):
        self.control = self._create_canvas(parent)
        self.set_tooltip()
        # Opt-in hook: a figure built with interactive hover/click wiring
        # (build_sequence_figure) stashes its wiring callback here since it
        # needs the canvas, which only exists after _create_canvas runs.
        on_ready = getattr(self.value, "_on_canvas_ready", None)
        if on_ready is not None:
            on_ready()

    def update_editor(self):
        pass

    def _create_canvas(self, parent):
        # Local imports: only pulled in when a window is actually created,
        # keeping module import cheap and headless-test friendly.
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg,
            NavigationToolbar2QT,
        )
        from pyface.qt import QtGui

        frame = QtGui.QWidget()
        canvas = FigureCanvasQTAgg(self.value)
        canvas.setParent(frame)
        toolbar = NavigationToolbar2QT(canvas, frame)

        layout = QtGui.QVBoxLayout(frame)
        layout.addWidget(toolbar)
        layout.addWidget(canvas)
        layout.setContentsMargins(0, 0, 0, 0)
        return frame


class MPLFigureEditor(BasicEditorFactory):
    """TraitsUI editor factory for a matplotlib Figure."""

    klass = _MPLFigureEditor


def compute_fov_bounds(vpar, cpar=None, cals=None):
    """Compute measurement-volume (field-of-view) axis limits.

    X and Z come straight from the criteria/volume parameters
    (``X_lay``, ``Zmin_lay``/``Zmax_lay``). The criteria have no Y extent, so
    Y is derived from ``volumedimension`` (ray-traced imaged volume) when the
    control params and calibrations are available; otherwise it falls back to
    the X span (assume a roughly square field of view).

    Args:
        vpar: VolumePar (or wrapper exposing ``_vpar``).
        cpar: ControlPar (or wrapper), optional — needed for the Y extent.
        cals: list of Calibration objects, optional — needed for the Y extent.

    Returns:
        ((xmin, xmax), (ymin, ymax), (zmin, zmax)).
    """
    v = getattr(vpar, "_vpar", vpar)
    x_lay = np.asarray(v.X_lay, dtype=float)
    z_vals = np.concatenate(
        [np.asarray(v.Zmin_lay, dtype=float), np.asarray(v.Zmax_lay, dtype=float)]
    )
    xlim = (float(x_lay.min()), float(x_lay.max()))
    zlim = (float(z_vals.min()), float(z_vals.max()))

    ylim = None
    if cpar is not None and cals is not None:
        try:
            from openptv2.algorithms.multimed import volumedimension

            cp = getattr(cpar, "_cpar", cpar)
            cl = [getattr(c, "_cal", c) for c in cals]
            _, _, ymax, ymin, _, _ = volumedimension(v, cp, cl)
            ylim = (float(min(ymin, ymax)), float(max(ymin, ymax)))
        except Exception:
            ylim = None
    if ylim is None:
        ylim = xlim  # ponytail: no Y criterion -> assume square FOV
    return (xlim, ylim, zlim)


def _draw_fov_box(ax, x0, x1, y0, y1, z0, z1):
    """Draw the 12 edges of the measurement-volume cuboid on a 3D axes."""
    edges = []
    # 4 edges along x, 4 along y, 4 along z
    for y in (y0, y1):
        for z in (z0, z1):
            edges.append(((x0, x1), (y, y), (z, z)))
    for x in (x0, x1):
        for z in (z0, z1):
            edges.append(((x, x), (y0, y1), (z, z)))
    for x in (x0, x1):
        for y in (y0, y1):
            edges.append(((x, x), (y, y), (z0, z1)))

    first = True
    for xs, ys, zs in edges:
        ax.plot(
            xs,
            ys,
            zs,
            color="#d1495b",
            lw=1.2,
            alpha=0.8,
            label="field of view" if first else None,
        )
        first = False


def build_3d_figure(points_xyz: np.ndarray, frame, bounds=None) -> Figure:
    """Build a 3D scatter Figure from an (N, 3) array of metric positions.

    Pure function (no window): safe to call under the Agg backend in tests.

    Args:
        points_xyz: (N, 3) array of x, y, z metric coordinates (mm).
        frame: frame identifier, shown in the title.
        bounds: optional ((xmin, xmax), (ymin, ymax), (zmin, zmax)) axis
            limits — the measurement field of view. When given, axes are
            clamped to it so out-of-volume outliers don't squash the view.

    Returns:
        A matplotlib Figure containing a single 3D axes.
    """
    points_xyz = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    n = points_xyz.shape[0]

    fig = Figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    if n > 0:
        x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
        sc = ax.scatter(x, y, z, c=z, cmap="viridis", s=12, depthshade=True)
        fig.colorbar(sc, ax=ax, shrink=0.6, label="z (mm)")

    if bounds is not None:
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = bounds
        _draw_fov_box(ax, xlo, xhi, ylo, yhi, zlo, zhi)
        # Pad the axes a little beyond the box so its edges are clearly
        # visible rather than flush against the axis planes.
        mx = (xhi - xlo) * 0.05 or 1.0
        my = (yhi - ylo) * 0.05 or 1.0
        mz = (zhi - zlo) * 0.05 or 1.0
        ax.set_xlim(xlo - mx, xhi + mx)
        ax.set_ylim(ylo - my, yhi + my)
        ax.set_zlim(zlo - mz, zhi + mz)
        ax.legend(loc="upper left", fontsize=8)

    # Equal physical aspect: without this mpl stretches each axis to fill the
    # plot cube independently, which visually exaggerates z in shallow volumes.
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    zlo, zhi = ax.get_zlim()
    ax.set_box_aspect((xhi - xlo or 1.0, yhi - ylo or 1.0, zhi - zlo or 1.0))

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title(f"3D positions — frame {frame} ({n} particles)")
    return fig


def _project_to_pixels(ax, xyz: np.ndarray) -> np.ndarray:
    """Project (N, 3) 3D data points to (N, 2) screen/pixel coordinates for
    the current view angle -- the standard trick for hit-testing on a
    rotatable 3D axes, since mouse events only carry pixel coords."""
    from mpl_toolkits.mplot3d import proj3d

    xs2d, ys2d, _ = proj3d.proj_transform(xyz[:, 0], xyz[:, 1], xyz[:, 2], ax.get_proj())
    return ax.transData.transform(np.column_stack([xs2d, ys2d]))


def _wire_measurement_events(fig: Figure, ax, all_xyz: np.ndarray, all_frame: np.ndarray) -> None:
    """Hover to identify a dot (frame + xyz shown top-left); click to select
    up to 4 dots (highlighted + numbered) and print running
    displacement/velocity/(from the 3rd point on) acceleration between
    consecutively selected points, using the real frame-number gaps."""
    state = {"selected": []}  # list of (frame, x, y, z)
    hover_text = ax.text2D(
        0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )
    picked_scatter = ax.scatter([], [], [], s=80, facecolors="none", edgecolors="red", linewidths=2)

    def _nearest(event):
        if event.x is None or event.y is None or len(all_xyz) == 0:
            return None
        px = _project_to_pixels(ax, all_xyz)
        d = np.hypot(px[:, 0] - event.x, px[:, 1] - event.y)
        idx = int(np.argmin(d))
        return idx if d[idx] < 15 else None

    def on_move(event):
        if event.inaxes is not ax:
            return
        idx = _nearest(event)
        if idx is None:
            return
        x, y, z = all_xyz[idx]
        hover_text.set_text(f"frame {int(all_frame[idx])}: ({x:.2f}, {y:.2f}, {z:.2f}) mm")
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is not ax:
            return
        idx = _nearest(event)
        if idx is None:
            return
        point = (int(all_frame[idx]), *all_xyz[idx].tolist())
        state["selected"].append(point)
        sel = state["selected"]
        pts = np.array([[p[1], p[2], p[3]] for p in sel])
        picked_scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        fig.canvas.draw_idle()

        n = len(sel)
        print(f"[3D measure] point {n}: frame {point[0]} = "
              f"({point[1]:.3f}, {point[2]:.3f}, {point[3]:.3f}) mm")
        if n >= 2:
            f1, *p1 = sel[-2]
            f2, *p2 = sel[-1]
            p1, p2 = np.array(p1), np.array(p2)
            dist = float(np.linalg.norm(p2 - p1))
            dframe = f2 - f1
            print(f"  displacement (pt{n - 1}->pt{n}): {dist:.4f} mm over {dframe} frame(s)"
                  + (f"  ->  velocity {dist / dframe:.4f} mm/frame" if dframe else ""))
        if n >= 3:
            f0, *p0 = sel[-3]
            f1b, *p1b = sel[-2]
            f2b, *p2b = sel[-1]
            p0, p1b, p2b = np.array(p0), np.array(p1b), np.array(p2b)
            dt1, dt2 = f1b - f0, f2b - f1b
            if dt1 and dt2:
                v1 = (p1b - p0) / dt1
                v2 = (p2b - p1b) / dt2
                accel = np.linalg.norm(v2 - v1) / ((dt1 + dt2) / 2.0)
                print(f"  acceleration (pt{n - 2}->pt{n - 1}->pt{n}): {accel:.4f} mm/frame^2")
        if n >= 4:
            print("  [4 points selected -- press 'c' to clear and start a new group]")

    def on_key(event):
        if event.key == "c":
            state["selected"] = []
            picked_scatter._offsets3d = ([], [], [])
            fig.canvas.draw_idle()
            print("[3D measure] selection cleared")

    def _connect():
        fig.canvas.mpl_connect("motion_notify_event", on_move)
        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)

    fig._on_canvas_ready = _connect


def build_sequence_figure(frames_points: dict, bounds=None) -> Figure:
    """Build an interactive multi-frame 3D scatter: one color per frame
    (viridis over frame order), hover to identify a dot, click to select up
    to 4 and see running displacement/velocity/acceleration printed (see
    _wire_measurement_events). Pure function except for the deferred event
    wiring, which is stashed on the Figure and fired once its canvas exists
    (see _MPLFigureEditor.init).

    Args:
        frames_points: {frame_number: (N, 3) array of xyz mm}, in display order.
        bounds: optional field-of-view axis limits, same as build_3d_figure.
    """
    fig = Figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    frames = sorted(frames_points)
    all_xyz_parts, all_frame_parts = [], []
    if frames:
        cmap = matplotlib.colormaps["viridis"]
        for i, f in enumerate(frames):
            pts = np.asarray(frames_points[f], dtype=float).reshape(-1, 3)
            if pts.shape[0] == 0:
                continue
            color = cmap(i / max(len(frames) - 1, 1))
            ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                color=color, s=10, depthshade=True,
                label=f"frame {f}" if len(frames) <= 12 else None,
            )
            all_xyz_parts.append(pts)
            all_frame_parts.append(np.full(pts.shape[0], f))

    all_xyz = np.concatenate(all_xyz_parts) if all_xyz_parts else np.empty((0, 3))
    all_frame = np.concatenate(all_frame_parts) if all_frame_parts else np.empty((0,))

    if bounds is not None:
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = bounds
        _draw_fov_box(ax, xlo, xhi, ylo, yhi, zlo, zhi)
        mx = (xhi - xlo) * 0.05 or 1.0
        my = (yhi - ylo) * 0.05 or 1.0
        mz = (zhi - zlo) * 0.05 or 1.0
        ax.set_xlim(xlo - mx, xhi + mx)
        ax.set_ylim(ylo - my, yhi + my)
        ax.set_zlim(zlo - mz, zhi + mz)

    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    zlo, zhi = ax.get_zlim()
    ax.set_box_aspect((xhi - xlo or 1.0, yhi - ylo or 1.0, zhi - zlo or 1.0))

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    n_total = int(all_xyz.shape[0])
    ax.set_title(
        f"3D positions — frames {frames[0]}..{frames[-1]} ({n_total} points)"
        if frames else "3D positions — no data"
    )
    if len(frames) <= 12 and frames:
        ax.legend(loc="upper left", fontsize=7)

    fig.text(
        0.02, 0.02,
        "Hover: identify a dot. Click: select (up to 4) for displacement/"
        "velocity/acceleration, printed to the terminal. 'c': clear selection.",
        fontsize=7, color="#555555",
    )
    _wire_measurement_events(fig, ax, all_xyz, all_frame)
    # The measurement wiring always adds one (empty when unselected) scatter
    # collection for the picked-point highlight, so axes.collections is never
    # truly empty -- stash the real point count for callers' "no data" checks.
    fig._n_points = n_total
    return fig


class Plot3DPositions(HasTraits):
    """TraitsUI window hosting the embedded 3D matplotlib scatter."""

    figure = Instance(Figure)

    traits_view = View(
        Item("figure", editor=MPLFigureEditor(), show_label=False),
        title="Visualize 3D positions",
        resizable=True,
        width=900,
        height=700,
    )


def _read_positions(rt_is_path: Path, frame: Optional[int] = None) -> np.ndarray:
    """Read an rt_is file or Zarr store and return its (N, 3) metric xyz positions.

    A file that exists but holds zero particles is returned as an empty
    (0, 3) array. Missing/unreadable files propagate their OSError.
    """
    if not rt_is_path.exists():
        # Fallback to the unified RunStore for this run.
        if frame is not None:
            from openptv2.storage import RunStore, find_existing_store

            # The rt_is file's parent may be res/, the experiment root, or
            # the store itself -- let find_existing_store disambiguate.
            zarr_path = find_existing_store(rt_is_path.parent) or (
                find_existing_store(rt_is_path.parent.parent)
                if rt_is_path.parent.name == "res"
                else None
            )
            if zarr_path is not None:
                try:
                    store = RunStore(zarr_path, mode="r")
                    if store.has_correspondences(frame):
                        pos_3d, _ = store.read_correspondences(frame)
                        return np.asarray(pos_3d, dtype=float)
                except Exception:
                    pass

    try:
        rows = ptv.read_rt_is_file(str(rt_is_path))  # [[x, y, z, p0..p3], ...]
    except ValueError:
        # read_rt_is_file raises ValueError when the header count is 0.
        return np.empty((0, 3), dtype=float)
    if not rows:
        return np.empty((0, 3), dtype=float)
    return np.asarray(rows, dtype=float)[:, :3]


def create_3d_positions_panel(rt_is_path, frame, bounds=None) -> Plot3DPositions:
    """Read an rt_is file and return a ready-to-show Plot3DPositions window.

    Args:
        rt_is_path: path to the rt_is.<frame> file.
        frame: frame identifier for the title.
        bounds: optional field-of-view axis limits (see build_3d_figure).
    """
    points = _read_positions(Path(rt_is_path), frame=frame)
    return Plot3DPositions(figure=build_3d_figure(points, frame, bounds=bounds))


def read_positions_sequence(exp_path, first: int, last: int, max_frames: int = 50) -> dict:
    """Read per-frame 3D positions (rt_is file, falling back to the store)
    across a frame range, same source as _read_positions per frame. Caps at
    max_frames (keeping the first max_frames), matching
    plot_3d_trajectories's own clamp -- so both stay bounded for a
    responsive interactive window."""
    exp_path = Path(exp_path)
    res_dir = exp_path / "res"
    last_clamped = min(last, first + max_frames - 1)

    frames_points = {}
    for frame in range(first, last_clamped + 1):
        pts = _read_positions(res_dir / f"rt_is.{frame}", frame=frame)
        if pts.shape[0] > 0:
            frames_points[frame] = pts
    return frames_points


def create_3d_positions_sequence_panel(exp_path, first: int, last: int, bounds=None, max_frames: int = 50) -> Plot3DPositions:
    """Read positions across a frame range and return a ready-to-show
    Plot3DPositions window with the interactive multi-frame scatter
    (build_sequence_figure): hover to identify a dot, click to measure
    displacement/velocity/acceleration between selected dots."""
    frames_points = read_positions_sequence(exp_path, first, last, max_frames=max_frames)
    return Plot3DPositions(figure=build_sequence_figure(frames_points, bounds=bounds))
