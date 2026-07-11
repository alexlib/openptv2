"""Calibration as a separate Tk application window.

Scaffolds the full button set of the current Traits calibration GUI in its own
Toplevel with a camera grid. Buttons are wired to status for now; each will be
connected to the existing ptv/orientation calls in the calibration slice
(Task 2.2), reusing the already-fixed residual-overlay logic (PR #19).
"""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np

from .events import EventBus, Status
from .widgets import MplImageView

# The action buttons mirror the current calibration GUI's left column.
CAL_ACTIONS = [
    "Load images/parameters",
    "Detection",
    "Manual orient.",
    "Orient. with file",
    "Show initial guess",
    "Sortgrid",
    "Raw orientation",
    "Fine tuning",
    "Orientation from dumbbell",
    "Restore ori files",
    "Edit calibration parameters",
    "Edit ori files",
    "Edit addpar files",
    "Orientation with particles",
]


class CalibrationWindow:
    def __init__(self, parent, pm, dataset_dir: Path, bus: EventBus | None = None,
                 palette: dict | None = None):
        self.pm = pm
        self.dataset_dir = Path(dataset_dir)
        self.bus = bus or EventBus()
        self.palette = palette
        self.num_cams = int(pm.parameters.get("num_cams", 4) or 4)
        self.views: dict[int, MplImageView] = {}

        self.top = tk.Toplevel(parent)
        self.top.title("Calibration")
        self.top.geometry("1200x820")

        left = ttk.Frame(self.top)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=4)
        self.buttons: dict[str, ttk.Button] = {}
        for label in CAL_ACTIONS:
            b = ttk.Button(left, text=label,
                           command=lambda lbl=label: self._action(lbl))
            b.pack(side=tk.TOP, fill=tk.X, pady=1)
            self.buttons[label] = b

        grid = ttk.Frame(self.top)
        grid.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        for cam in range(self.num_cams):
            v = MplImageView(grid, cam=cam, bus=self.bus, palette=self.palette)
            img = self._cal_image(cam)
            v.set_image(img)
            r, c = divmod(cam, 2)
            v.grid(row=r, column=c, sticky="nsew", padx=2, pady=2)
            self.views[cam] = v
        for i in range((self.num_cams + 1) // 2):
            grid.rowconfigure(i, weight=1)
        for i in range(2):
            grid.columnconfigure(i, weight=1)

        self.status = ttk.Label(self.top, text="calibration ready",
                                anchor="w", relief=tk.SUNKEN)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def _cal_image(self, cam: int) -> np.ndarray:
        cand = self.dataset_dir / f"cal/cam{cam+1}.tif"
        if cand.exists():
            try:
                import imageio.v3 as iio
                img = np.asarray(iio.imread(cand))
                return img[..., 0] if img.ndim > 2 else img
            except Exception:
                pass
        return np.zeros((64, 64), dtype=np.uint8)

    def _action(self, label: str) -> None:
        msg = f"calibration: '{label}' — connect to ptv/orientation in Task 2.2"
        self.status.config(text=msg)
        self.bus.publish(Status(msg))
