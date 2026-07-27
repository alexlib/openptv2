"""Interactive 3D visualization of rt_is particle positions using PyVista.

NOTE: This branch fails with PyVista on Windows due to PySide6 / pyvistaqt
QVTKRenderWindowInteractor integration issues with TraitsUI windows.

Opens a TraitsUI window hosting an embedded, mouse-rotatable PyVista
3D visualization of the 3D positions stored in an rt_is.<frame> file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
from traits.api import HasTraits, Instance
from traitsui.api import Item, View
from traitsui.basic_editor_factory import BasicEditorFactory
from traitsui.qt.editor import Editor

from . import ptv


def _get_parent_widget(parent):
    from PySide6.QtWidgets import QWidget

    if isinstance(parent, QWidget):
        return parent
    if hasattr(parent, "parentWidget") and callable(parent.parentWidget):
        p = parent.parentWidget()
        if isinstance(p, QWidget):
            return p
    if hasattr(parent, "parent") and callable(parent.parent):
        p = parent.parent()
        if isinstance(p, QWidget):
            return p
    return None


class _PyVistaEditor(Editor):
    """Embeds a PyVista QtInteractor inside a TraitsUI Item."""

    scrollable = True

    def init(self, parent):
        from pyvistaqt import QtInteractor

        parent_widget = _get_parent_widget(parent)
        if isinstance(self.value, QtInteractor):
            self.control = self.value
        else:
            interactor = QtInteractor(parent=parent_widget)
            if callable(self.value):
                self.value(interactor)
            self.control = interactor
        self.set_tooltip()

    def update_editor(self):
        pass


class PyVistaEditor(BasicEditorFactory):
    """TraitsUI editor factory for a PyVista QtInteractor."""

    klass = _PyVistaEditor


def compute_fov_bounds(vpar, cpar=None, cals=None):
    """Compute measurement-volume (field-of-view) axis limits.

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
        ylim = xlim  # fallback: assume square field of view
    return (xlim, ylim, zlim)


def build_3d_positions_plotter(
    points_xyz: np.ndarray, frame, bounds=None, plotter: pv.Plotter | None = None
) -> pv.Plotter:
    """Build a PyVista Plotter displaying 3D particle positions (tracers).

    Pure function: safe to call off-screen in tests or headless environments.

    Args:
        points_xyz: (N, 3) array of x, y, z metric coordinates (mm).
        frame: frame identifier, shown in title.
        bounds: optional ((xmin, xmax), (ymin, ymax), (zmin, zmax)) FOV limits.
        plotter: optional existing PyVista Plotter / QtInteractor instance.

    Returns:
        The populated PyVista Plotter instance.
    """
    if plotter is None:
        plotter = pv.Plotter(off_screen=True)

    points_xyz = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    n = points_xyz.shape[0]

    if n > 0:
        cloud = pv.PolyData(points_xyz)
        plotter.add_mesh(
            cloud,
            scalars=points_xyz[:, 2],
            cmap="viridis",
            render_points_as_spheres=True,
            point_size=10,
            scalar_bar_args={"title": "z (mm)"},
            label="particles",
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
    plotter.add_title(f"3D positions — frame {frame} ({n} particles)", font_size=12)
    plotter.reset_camera()
    return plotter


class Plot3DPositions(HasTraits):
    """TraitsUI window hosting the embedded 3D PyVista plot."""

    plotter_builder = Instance(object)

    traits_view = View(
        Item("plotter_builder", editor=PyVistaEditor(), show_label=False),
        title="Visualize 3D positions",
        resizable=True,
        width=900,
        height=700,
    )


def _read_positions(rt_is_path: Path) -> np.ndarray:
    """Read an rt_is file and return its (N, 3) metric xyz positions."""
    try:
        rows = ptv.read_rt_is_file(str(rt_is_path))
    except ValueError:
        return np.empty((0, 3), dtype=float)
    if not rows:
        return np.empty((0, 3), dtype=float)
    return np.asarray(rows, dtype=float)[:, :3]


def create_3d_positions_panel(rt_is_path, frame, bounds=None) -> Plot3DPositions:
    """Read an rt_is file and return a ready-to-show Plot3DPositions window."""
    points = _read_positions(Path(rt_is_path))

    def builder(interactor):
        build_3d_positions_plotter(points, frame, bounds=bounds, plotter=interactor)

    panel = Plot3DPositions(plotter_builder=builder)
    panel.points = points
    return panel
