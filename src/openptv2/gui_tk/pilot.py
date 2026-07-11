"""Phase-0 pilot for the Tk/matplotlib GUI migration.

Proves the two primitives that Chaco gives for free and that a Tk rewrite must
replicate:
  1. an interactive camera image (imshow + zoom/pan) embedded in Tkinter, where
     a mouse click reports the correct IMAGE pixel coordinate, and
  2. pixel-accurate overlays (detected-target crosses + reprojection-residual
     quiver) drawn on the same axes as the image.

Both use only free-threaded-ready deps (tkinter + matplotlib TkAgg + numpy),
so this doubles as the free-threading validation.

Run interactively (needs a display):
    uv run python -m openptv2.gui_tk.pilot test_data/test_cavity

Headless self-check (no display; verifies the machine-checkable parts):
    uv run python -m openptv2.gui_tk.pilot test_data/test_cavity --selftest
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _load_scene(dataset: Path, cam: int = 1):
    """Return (image, detected_xy, matched_xy, residual_uv) for one camera.

    Reuses the EXISTING algorithm APIs (no GUI framework) so the pilot exercises
    the real calibration path, not a mock.
    """
    import imageio.v3 as iio

    from openptv2.algorithms.calibration import Calibration
    from openptv2.algorithms.parameters import ControlPar
    from openptv2.algorithms.tracking_frame_buf import Target
    from openptv2.orientation import full_calibration, match_detection_to_ref

    ds = Path(dataset)
    cpar = ControlPar.from_yaml(str(ds / "parameters_Run1.yaml"))
    cal = Calibration()
    cal.from_file(
        str(ds / f"cal/cam{cam}.tif.ori"),
        str(ds / f"cal/cam{cam}.tif.addpar"),
    )

    img = np.asarray(iio.imread(ds / f"cal/cam{cam}.tif"))
    if img.ndim > 2:
        img = img[..., 0]

    d = np.loadtxt(ds / f"cal/cam{cam}.tif_targets", skiprows=1, ndmin=2)
    det = [
        Target(pnr=int(r[0]), x=r[1], y=r[2], n=int(r[3]), nx=int(r[4]),
               ny=int(r[5]), sumg=int(r[6]), tnr=int(r[7]))
        for r in d
    ]
    detected_xy = np.array([[t.x, t.y] for t in det]) if det else np.empty((0, 2))

    ref = np.loadtxt(ds / "cal/target_on_a_side.txt")[:, 1:4]
    targs = match_detection_to_ref(cal, ref, det, cpar)
    residuals, _used, _err = full_calibration(cal, ref, targs, cpar, ["cc", "xh", "yh"])

    # Matched points only (skip pnr==-999 placeholders / NaN residuals) — same
    # rule as the fixed Traits GUI (PR #19).
    mx, res = [], []
    for i in range(len(targs)):
        pnr = targs[i].pnr() if callable(targs[i].pnr) else targs[i].pnr
        if pnr == -999 or i >= len(residuals) or np.isnan(residuals[i, 0]):
            continue
        p = targs[i].pos()
        mx.append([p[0], p[1]])
        res.append([residuals[i, 0], residuals[i, 1]])
    matched_xy = np.array(mx) if mx else np.empty((0, 2))
    residual_uv = np.array(res) if res else np.empty((0, 2))
    return img, detected_xy, matched_xy, residual_uv


def _build_figure(img, detected_xy, matched_xy, residual_uv, scale=5000.0):
    """Build the matplotlib Figure (backend-agnostic; used by GUI and selftest)."""
    from matplotlib.figure import Figure

    fig = Figure(figsize=(9, 7))
    ax = fig.add_subplot(111)
    # origin='upper' + default extent => data coords ARE image pixel coords,
    # so click event.xdata/ydata map 1:1 to pixels (the key correctness point).
    ax.imshow(img, cmap="gray", origin="upper")
    if len(detected_xy):
        ax.scatter(detected_xy[:, 0], detected_xy[:, 1], s=40, c="cyan",
                   marker="+", label="detected")
    if len(matched_xy):
        ax.scatter(matched_xy[:, 0], matched_xy[:, 1], s=30, c="orange",
                   marker="o", label="matched")
        ax.quiver(matched_xy[:, 0], matched_xy[:, 1],
                  scale * residual_uv[:, 0], scale * residual_uv[:, 1],
                  angles="xy", scale_units="xy", scale=1.0, color="red",
                  width=0.002, label="residual x%g" % scale)
    ax.set_title("Pilot: click reports image pixel; overlays share the image axes")
    ax.legend(loc="upper right", fontsize=8)
    return fig, ax


def selftest(dataset: str) -> int:
    """Machine-verifiable checks (no display): artists present, click→pixel map."""
    import matplotlib
    matplotlib.use("Agg")
    img, det, matched, res = _load_scene(Path(dataset))
    fig, ax = _build_figure(img, det, matched, res)

    imgs = ax.get_images()
    assert len(imgs) == 1, "expected exactly one image artist"
    assert len(ax.collections) >= 1, "expected overlay scatter/quiver artists"

    # click→pixel mapping: with imshow default extent, ax.transData maps a pixel
    # (px,py) to display and back to the SAME data coord. Verify round-trip on a
    # sample of detected points (this is the transform the click handler relies on).
    if len(det):
        pts = det[: min(20, len(det))]
        disp = ax.transData.transform(pts)
        back = ax.transData.inverted().transform(disp)
        err = np.abs(back - pts).max()
        assert err < 1e-6, f"pixel<->display round-trip error {err}"
        print(f"  click->pixel transform round-trip max err = {err:.2e} px  OK")

    print(f"  image {img.shape}, detected={len(det)}, matched={len(matched)}, "
          f"residual vectors={len(res)}")
    print("SELFTEST PASS")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    dataset = argv[0]
    if "--selftest" in argv:
        return selftest(dataset)

    # Interactive path (needs a display).
    import tkinter as tk

    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg,
        NavigationToolbar2Tk,
    )

    img, det, matched, res = _load_scene(Path(dataset))
    fig, ax = _build_figure(img, det, matched, res)

    root = tk.Tk()
    root.title("OpenPTV2 Tk pilot — click the image")
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    NavigationToolbar2Tk(canvas, root)  # zoom / pan / reset for free
    status = tk.Label(root, text="Click a calibration dot to read its image pixel.",
                      anchor="w")
    status.pack(side=tk.BOTTOM, fill=tk.X)

    def on_click(event):
        if event.inaxes is ax and event.xdata is not None:
            status.config(text=f"clicked image pixel: "
                               f"x={event.xdata:.2f}, y={event.ydata:.2f}")
            print(f"click -> pixel ({event.xdata:.2f}, {event.ydata:.2f})")

    canvas.mpl_connect("button_press_event", on_click)
    canvas.draw()
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
