"""Embedded 3-D positions view (matplotlib Axes3D in a Tk Toplevel).

Reads rt_is.* from the dataset's res/ and shows the 3-D point cloud; rotatable
with the mouse. Replaces the current traitsui-hosted matplotlib 3-D plot with a
plain FigureCanvasTkAgg (free-threaded-ready).
"""
from __future__ import annotations

import glob
import tkinter as tk
from pathlib import Path

import numpy as np

from .events import EventBus


def _load_points(res_dir: Path) -> np.ndarray:
    pts = []
    for f in sorted(glob.glob(str(res_dir / "rt_is.*"))):
        try:
            d = np.loadtxt(f, skiprows=1, ndmin=2)
            if d.size and d.shape[1] >= 4:
                pts.append(d[:, 1:4])
        except Exception:
            continue
    return np.vstack(pts) if pts else np.empty((0, 3))


class View3DWindow:
    def __init__(self, parent, dataset_dir: Path, bus: EventBus | None = None):
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg,
            NavigationToolbar2Tk,
        )
        from matplotlib.figure import Figure

        self.top = tk.Toplevel(parent)
        self.top.title("3D positions")
        self.top.geometry("700x600")

        fig = Figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        pts = _load_points(Path(dataset_dir) / "res")
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"{len(pts)} points")

        canvas = FigureCanvasTkAgg(fig, master=self.top)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, self.top)
        canvas.draw()
        self.n_points = len(pts)
