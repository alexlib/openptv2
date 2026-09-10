"""Draw a 3D trajectory ROI as inclusion polygons on the XY/XZ/YZ projections.

Reads a run's 3D trajectory positions straight from its Zarr store (no per-
camera images, no tabs) and shows one dialog with the three projections
side by side. Draw a polygon on any subset of them with
``matplotlib.widgets.PolygonSelector`` (click to add vertices, close the
loop to finish, ``esc`` to reset it), then press "Save ROI" to write
``trajectory_roi.json`` in the working folder -- consumed later via
``openptv2.trajectory_roi.TrajectoryROI`` to filter trajectories that leave
the selected region in any drawn plane.

The figure is embedded in a native Qt dialog (``FigureCanvasQTAgg`` in a
``QDialog``) rather than opened via ``matplotlib.pyplot.show()``: this GUI
runs inside PySide6's own already-running Qt event loop (started by
pyface/traitsui), and pyplot's ``show()`` starts a second, competing event
loop on top of it -- the figure appears but never receives mouse clicks and
button presses go nowhere. A modal ``QDialog.exec()`` nests cleanly inside
the host's loop instead.
"""

from pathlib import Path

import numpy as np

from openptv2.storage import RunStore, find_existing_store
from openptv2.trajectory_roi import PLANES, TrajectoryROI

ROI_FILENAME = "trajectory_roi.json"

_TITLES = {"xy_polygon": "XY", "xz_polygon": "XZ", "yz_polygon": "YZ"}


def load_trajectory_positions(working_folder: Path | str) -> np.ndarray:
    """Load all (N, 3) trajectory positions for the run in ``working_folder``."""
    store_path = find_existing_store(working_folder)
    if store_path is None:
        raise FileNotFoundError(
            f"No run store (res/run.zarr) found under {working_folder}"
        )
    return RunStore(store_path, mode="r").trajectories()["pos"]


def draw_trajectory_roi(working_folder: Path | str) -> None:
    """Open the interactive ROI dialog for the run in ``working_folder``."""
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from matplotlib.widgets import PolygonSelector
    from PySide6.QtWidgets import (
        QApplication,
        QDialog,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QVBoxLayout,
    )

    working_folder = Path(working_folder)
    positions = load_trajectory_positions(working_folder)
    roi_path = working_folder / ROI_FILENAME
    saved_roi = TrajectoryROI.from_json(roi_path) if roi_path.exists() else TrajectoryROI()

    # A QApplication must exist before the first QWidget is constructed --
    # QApplication.instance() picks up the host app's existing one when
    # called from openptv2-gui; standalone use creates one here.
    app = QApplication.instance() or QApplication([])

    dialog = QDialog()
    dialog._app = app  # keep it alive on the dialog for standalone use
    dialog.setWindowTitle(f"Draw trajectory ROI -- {working_folder}")
    dialog.resize(1300, 550)
    layout = QVBoxLayout(dialog)

    figure = Figure(figsize=(15, 5))
    canvas = FigureCanvasQTAgg(figure)
    layout.addWidget(canvas)
    axes = figure.subplots(1, 3)

    polygons = {name: getattr(saved_roi, name) for name in PLANES}
    selectors = {}
    for ax, (name, (a, b)) in zip(axes, PLANES.items()):
        ax.scatter(positions[:, a], positions[:, b], s=2, alpha=0.15, color="#375a7f")
        ax.set_title(_TITLES[name])
        ax.set_aspect("equal", adjustable="box")

        def on_select(vertices, name=name):
            polygons[name] = np.asarray(vertices, dtype=float)

        selectors[name] = PolygonSelector(ax, on_select, useblit=True)
        if polygons[name] is not None:
            selectors[name].verts = list(map(tuple, polygons[name]))
    figure.tight_layout()

    status = QLabel(
        "Click to add vertices, click the first vertex to close a polygon, "
        "Esc to reset it. Then Save ROI."
    )
    layout.addWidget(status)

    buttons = QHBoxLayout()
    save_button = QPushButton("Save ROI")
    close_button = QPushButton("Close")
    buttons.addWidget(save_button)
    buttons.addWidget(close_button)
    layout.addLayout(buttons)

    def save():
        TrajectoryROI(**polygons).to_json(roi_path)
        status.setText(f"Saved: {roi_path}")

    save_button.clicked.connect(save)
    close_button.clicked.connect(dialog.accept)

    dialog.exec()


if __name__ == "__main__":
    import sys

    draw_trajectory_roi(sys.argv[1] if len(sys.argv) > 1 else ".")
