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

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title(f"3D positions — frame {frame} ({n} particles)")
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


def _read_positions(rt_is_path: Path) -> np.ndarray:
    """Read an rt_is file and return its (N, 3) metric xyz positions.

    A file that exists but holds zero particles is returned as an empty
    (0, 3) array. Missing/unreadable files propagate their OSError.
    """
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
    points = _read_positions(Path(rt_is_path))
    return Plot3DPositions(figure=build_3d_figure(points, frame, bounds=bounds))
