"""Draw a 3D trajectory ROI as inclusion polygons on the XY/XZ/YZ projections.

Reads a run's 3D trajectory positions straight from its Zarr store (no per-
camera images, no tabs) and shows one matplotlib figure with the three
projections side by side. Draw a polygon on any subset of them with
``matplotlib.widgets.PolygonSelector`` (click to add vertices, close the
loop to finish, ``esc`` to reset it), then press "Save ROI" to write
``trajectory_roi.json`` in the working folder -- consumed later via
``openptv2.trajectory_roi.TrajectoryROI`` to filter trajectories that leave
the selected region in any drawn plane.
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
    """Open the interactive ROI figure for the run in ``working_folder``."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, PolygonSelector

    working_folder = Path(working_folder)
    positions = load_trajectory_positions(working_folder)
    roi_path = working_folder / ROI_FILENAME
    saved_roi = TrajectoryROI.from_json(roi_path) if roi_path.exists() else TrajectoryROI()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
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

    def save(_event):
        TrajectoryROI(**polygons).to_json(roi_path)
        fig.suptitle(f"Saved: {roi_path}")
        fig.canvas.draw_idle()

    save_ax = fig.add_axes((0.45, 0.01, 0.1, 0.05))
    save_button = Button(save_ax, "Save ROI")
    save_button.on_clicked(save)

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    plt.show()


if __name__ == "__main__":
    import sys

    draw_trajectory_roi(sys.argv[1] if len(sys.argv) > 1 else ".")
