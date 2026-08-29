"""Interactive detection-parameter tuning GUI (marimo notebook).

Loads a dataset's ``parameters_*.yaml``, reads real frames via the shared
``openptv2.gui.ptv.read_frame_images`` loader (handles splitter/non-splitter/
zarr transparently), and overlays ``targ_rec`` detections live as you adjust
sliders. Each camera gets its own binarization threshold (``gvthres`` is a
per-camera array); the remaining detection parameters (discontinuity,
pixel-count bounds, extent bounds, min sum-grey) are shared across all
cameras, matching how ``targ_rec.par``/the YAML actually structure them.

Run:
    uv run --extra viz marimo edit src/openptv2/gui/detection_gui_nb.py

Then set the dataset path and frame in the UI (no CLI args -- this is an
edit-mode tuning tool, not a read-only viewer).
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Detection tuning GUI

    Point this at a dataset's YAML, pick a frame, and adjust thresholds —
    the overlay updates live. Per-camera grey-value threshold; shared
    discontinuity/size/brightness bounds across all cameras.
    """)
    return


@app.cell
def _(mo):
    yaml_path_ui = mo.ui.text(
        value=r"C:\Users\alex\Downloads\wp1_10_images\parameters_wp1.yaml",
        label="parameters_*.yaml path",
        full_width=True,
    )
    yaml_path_ui
    return (yaml_path_ui,)


@app.cell
def _(yaml_path_ui):
    from pathlib import Path

    from openptv2.gui.parameter_manager import ParameterManager
    from openptv2.gui.ptv import py_start_proc_c, read_frame_images

    pm = ParameterManager()
    pm.from_yaml(Path(yaml_path_ui.value))
    cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(pm)
    num_cams = cpar.num_cams
    return Path, num_cams, pm, read_frame_images, spar, tpar


@app.cell
def _(mo, spar):
    frame_ui = mo.ui.number(
        value=spar.first, start=spar.first, stop=spar.last, step=1, label="frame"
    )
    frame_ui
    return (frame_ui,)


@app.cell
def _(Path, frame_ui, num_cams, pm, read_frame_images, spar, yaml_path_ui):
    yaml_dir = Path(yaml_path_ui.value).parent
    absolute_img_base_name = [
        str(yaml_dir / name) if not Path(name).is_absolute() else name
        for name in spar.img_base_name
    ]
    images = read_frame_images(pm, absolute_img_base_name, num_cams, int(frame_ui.value))
    return (images,)


@app.cell
def _(gvthres_uis, mo, num_cams):

    cam_ui = mo.ui.radio(
        options=[f"cam{c + 1}" for c in range(num_cams)],
        value="cam1",
        label="select camera",
    )

    cam_idx = int(cam_ui.value.replace("cam", "")) - 1
    gvthres_uis[cam_idx]

    return cam_idx, cam_ui


@app.cell
def _(mo, tpar):

    # Crop region
    x_center_ui = mo.ui.slider(0, 512, value=256, step=1, label="x center")
    y_center_ui = mo.ui.slider(0, 512, value=256, step=1, label="y center")
    zoom_ui = mo.ui.slider(10, 512, value=200, step=10, label="zoom")

    # Detection parameters
    discont_ui = mo.ui.slider(1, 200, value=int(tpar.discont), label="discont")
    nnmin_ui = mo.ui.slider(1, 50, value=int(tpar.nnmin), label="nnmin")
    nnmax_ui = mo.ui.slider(1, 500, value=int(tpar.nnmax), label="nnmax")
    nxmin_ui = mo.ui.slider(1, 50, value=int(tpar.nxmin), label="nxmin")
    nxmax_ui = mo.ui.slider(1, 50, value=int(tpar.nxmax), label="nxmax")
    nymin_ui = mo.ui.slider(1, 50, value=int(tpar.nymin), label="nymin")
    nymax_ui = mo.ui.slider(1, 50, value=int(tpar.nymax), label="nymax")
    sumg_min_ui = mo.ui.slider(0, 500, value=int(tpar.sumg_min), label="sumg_min")

    mo.vstack([
        mo.hstack([x_center_ui, y_center_ui, zoom_ui]),
        mo.hstack([discont_ui, sumg_min_ui]),
        mo.hstack([nnmin_ui, nnmax_ui]),
        mo.hstack([nxmin_ui, nxmax_ui, nymin_ui, nymax_ui]),
    ])

    return (
        discont_ui,
        nnmax_ui,
        nnmin_ui,
        nxmax_ui,
        nxmin_ui,
        nymax_ui,
        nymin_ui,
        sumg_min_ui,
        x_center_ui,
        y_center_ui,
        zoom_ui,
    )


@app.cell
def _(
    cam_idx,
    cam_ui,
    discont_ui,
    gvthres_uis,
    images,
    nnmax_ui,
    nnmin_ui,
    np,
    nxmax_ui,
    nxmin_ui,
    nymax_ui,
    nymin_ui,
    plt,
    sumg_min_ui,
    x_center_ui,
    y_center_ui,
    zoom_ui,
):

    from matplotlib.patches import Rectangle

    from openptv2.algorithms.segmentation import targ_rec

    _ = gvthres_uis[cam_idx].value
    _ = discont_ui.value
    _ = nnmin_ui.value
    _ = nnmax_ui.value
    _ = nxmin_ui.value
    _ = nxmax_ui.value
    _ = nymin_ui.value
    _ = nymax_ui.value
    _ = sumg_min_ui.value
    _ = cam_ui.value
    _ = x_center_ui.value
    _ = y_center_ui.value
    _ = zoom_ui.value

    img = np.ascontiguousarray(images[cam_idx], dtype=np.uint8)
    h, w = img.shape[:2]

    hw = int(zoom_ui.value)
    x0 = max(0, int(x_center_ui.value) - hw)
    x1 = min(w, int(x_center_ui.value) + hw)
    y0 = max(0, int(y_center_ui.value) - hw)
    y1 = min(h, int(y_center_ui.value) + hw)

    targets = targ_rec(
        img,
        int(gvthres_uis[cam_idx].value),
        int(discont_ui.value),
        int(nnmin_ui.value),
        int(nnmax_ui.value),
        int(nxmin_ui.value),
        int(nxmax_ui.value),
        int(nymin_ui.value),
        int(nymax_ui.value),
        int(sumg_min_ui.value),
    )
    if len(targets) == 1 and targets[0].pnr == 1 and targets[0].x == 1:
        targets = []

    crop_targets = [t for t in targets if x0 <= t.x <= x1 and y0 <= t.y <= y1]
    count = len(crop_targets)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.imshow(img, cmap="gray", vmin=0, vmax=255)
    if targets:
        all_xs = [t.x for t in targets]
        all_ys = [t.y for t in targets]
        ax1.scatter(all_xs, all_ys, s=8, marker="x", color="red", linewidths=0.8)
    rect = Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor="cyan", facecolor="none")
    ax1.add_patch(rect)
    ax1.set_title(f"cam{cam_idx + 1}: {len(targets)} detections (gvthres={gvthres_uis[cam_idx].value})")
    ax1.axis("off")

    crop_img = img[y0:y1, x0:x1]
    ax2.imshow(crop_img, cmap="gray", vmin=0, vmax=255)
    if crop_targets:
        xs = [t.x - x0 for t in crop_targets]
        ys = [t.y - y0 for t in crop_targets]
        ax2.scatter(xs, ys, s=30, facecolors="none", edgecolors="red", linewidths=1)
    ax2.set_title(f"detail: {count} in region")
    ax2.axis("off")

    fig.tight_layout()
    fig

    return


@app.cell
def _(mo):
    mo.md("""
    **detection count for cam{cam_idx + 1}:** {count}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### Export tuned parameters

    Writes the current slider values back into `criteria.eps0`-sibling
    `targ_rec` block of a copy of the loaded YAML.
    """)
    return


@app.cell
def _(mo):
    export_path_ui = mo.ui.text(
        value="", label="save tuned yaml as (blank = don't save)", full_width=True
    )
    export_button = mo.ui.run_button(label="save")
    mo.hstack([export_path_ui, export_button])
    return export_button, export_path_ui


@app.cell
def _(
    Path,
    cam_idx,
    discont_ui,
    export_button,
    export_path_ui,
    gvthres_ui,
    mo,
    nnmax_ui,
    nnmin_ui,
    nxmax_ui,
    nxmin_ui,
    nymax_ui,
    nymin_ui,
    pm,
    sumg_min_ui,
    tpar,
    yaml_path_ui,
):

    if export_button.value and export_path_ui.value:
        pm.parameters.setdefault("targ_rec", {})
        # Update only the selected camera's threshold
        gvthres_list = list(pm.parameters["targ_rec"].get("gvthres", tpar.gvthres))
        gvthres_list[cam_idx] = int(gvthres_ui.value)
        pm.parameters["targ_rec"]["gvthres"] = gvthres_list
        pm.parameters["targ_rec"]["disco"] = int(discont_ui.value)
        pm.parameters["targ_rec"]["nnmin"] = int(nnmin_ui.value)
        pm.parameters["targ_rec"]["nnmax"] = int(nnmax_ui.value)
        pm.parameters["targ_rec"]["nxmin"] = int(nxmin_ui.value)
        pm.parameters["targ_rec"]["nxmax"] = int(nxmax_ui.value)
        pm.parameters["targ_rec"]["nymin"] = int(nymin_ui.value)
        pm.parameters["targ_rec"]["nymax"] = int(nymax_ui.value)
        pm.parameters["targ_rec"]["sumg_min"] = int(sumg_min_ui.value)
        out = Path(export_path_ui.value)
        pm.to_yaml(out)
        result = mo.md(f"saved to {out} (source: {yaml_path_ui.value})")
    else:
        result = mo.md("*(not saved)*")
    result
    return


@app.cell(hide_code=True)
def _(cam_ui):

    # Compute camera index from selector
    cam_idx = int(cam_ui.value.replace("cam", "")) - 1
    cam_idx
    return (cam_idx,)


@app.cell(hide_code=True)
def _(mo, num_cams, tpar):

    gvthres_uis = [
        mo.ui.slider(0, 255, value=int(tpar.gvthres[c]), label=f"cam{c+1} gvthres")
        for c in range(num_cams)
    ]

    return (gvthres_uis,)


if __name__ == "__main__":
    app.run()
