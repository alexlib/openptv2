"""Reusable Tk widgets for the GUI: an interactive matplotlib camera view and a
detachable panel wrapper.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

import numpy as np

from .events import CameraClick, EventBus


class MplImageView(ttk.Frame):
    """A camera image embedded in Tk via matplotlib, with zoom/pan, overlays and
    click reporting (image-pixel coordinates).

    Clicks publish a :class:`CameraClick` on the bus (so other views can react —
    e.g. draw epipolar lines) and/or call an ``on_click`` callback.
    """

    def __init__(self, master, cam: int = 0, bus: EventBus | None = None,
                 on_click: Callable[[float, float, int], None] | None = None):
        super().__init__(master)
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg,
            NavigationToolbar2Tk,
        )
        from matplotlib.figure import Figure

        self.cam = cam
        self.bus = bus
        self._on_click = on_click

        self._fig = Figure(figsize=(4, 3))
        self._ax = self._fig.add_subplot(111)
        self._ax.set_axis_off()
        self._img_artist = None
        self._overlays: list = []

        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self._canvas, self, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self._canvas.mpl_connect("button_press_event", self._handle_click)

    # --- image + overlays -------------------------------------------------- #

    def set_image(self, img: np.ndarray) -> None:
        img = np.asarray(img)
        if img.ndim > 2:
            img = img[..., 0]
        if self._img_artist is None:
            self._img_artist = self._ax.imshow(img, cmap="gray", origin="upper")
        else:
            self._img_artist.set_data(img)
            self._img_artist.set_clim(img.min(), img.max())
        self._canvas.draw_idle()

    def clear_overlays(self) -> None:
        for artist in self._overlays:
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                pass
        self._overlays = []
        self._canvas.draw_idle()

    def add_points(self, xy, color="cyan", marker="+", size=40):
        xy = np.asarray(xy)
        if len(xy):
            self._overlays.append(
                self._ax.scatter(xy[:, 0], xy[:, 1], s=size, c=color, marker=marker)
            )
        self._canvas.draw_idle()

    def add_quiver(self, xy, uv, scale=1.0, color="red"):
        xy, uv = np.asarray(xy), np.asarray(uv)
        if len(xy):
            self._overlays.append(self._ax.quiver(
                xy[:, 0], xy[:, 1], scale * uv[:, 0], scale * uv[:, 1],
                angles="xy", scale_units="xy", scale=1.0, color=color, width=0.003,
            ))
        self._canvas.draw_idle()

    def add_line(self, x0, y0, x1, y1, color="yellow"):
        """Draw a line segment (used for epipolar lines from other views)."""
        (ln,) = self._ax.plot([x0, x1], [y0, y1], "-", color=color, linewidth=0.8)
        self._overlays.append(ln)
        self._canvas.draw_idle()

    def draw(self) -> None:
        self._canvas.draw_idle()

    # --- click handling ---------------------------------------------------- #

    def _handle_click(self, event) -> None:
        if event.inaxes is not self._ax or event.xdata is None:
            return
        x, y, btn = float(event.xdata), float(event.ydata), event.button or 1
        if self.bus is not None:
            self.bus.publish(CameraClick(cam=self.cam, x=x, y=y, button=btn))
        if self._on_click is not None:
            self._on_click(x, y, btn)


class DetachablePanel(ttk.Frame):
    """A titled frame whose content can pop out into its own Toplevel and back."""

    def __init__(self, master, title: str, build: Callable[[tk.Widget], tk.Widget]):
        super().__init__(master)
        self._title = title
        self._build = build
        self._top: tk.Toplevel | None = None

        bar = ttk.Frame(self)
        bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(bar, text=title).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="⧉", width=3, command=self.detach).pack(side=tk.RIGHT)

        self._body = ttk.Frame(self)
        self._body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._content = build(self._body)
        self._content.pack(fill=tk.BOTH, expand=True)

    def detach(self) -> None:
        if self._top is not None:
            return
        self._top = tk.Toplevel(self)
        self._top.title(self._title)
        self._content.pack_forget()
        self._content = self._build(self._top)
        self._content.pack(in_=self._top, fill=tk.BOTH, expand=True)
        self._top.protocol("WM_DELETE_WINDOW", self._reattach)

    def _reattach(self) -> None:
        if self._top is None:
            return
        self._top.destroy()
        self._top = None
        self._content = self._build(self._body)
        self._content.pack(in_=self._body, fill=tk.BOTH, expand=True)
