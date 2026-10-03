# /// script
# dependencies = [
#     "marimo",
#     "numpy",
#     "matplotlib",
#     "openptv2==0.5.15",
#     "pyyaml==6.0.3",
# ]
# requires-python = ">=3.11"
# ///
"""Interactive bubble-detection tuning for the Ilmenau rigs.

Background on/off, background pedigree (frames/method), background image
before subtraction, demo frame with vs without background, and detection
with vs without background — per rig, with one-click write-back of the
`bubble_detection:` block into that rig's parameters_Run1.yaml.

Run:
    uv run --project /Users/alex/Documents/Github/openptv2 marimo edit \
        /Users/alex/Downloads/Ilmenau/bubble_detection_tuning_marimo.py
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return Path, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Bubble-detection tuning (Ilmenau rigs 1–4 / 5–8)

    Tune the `bubble_sequence` detector from `openptv2`: local-contrast
    window, z-threshold, glare merging — with and without static-background
    subtraction (barrel walls, seams, sensor pattern). Parameters mirror
    `bubble_detection:` in `parameters_Run1.yaml` exactly, so what works
    here transfers to batch runs 1:1.
    """)
    return


@app.cell
def _(mo):
    rig_select = mo.ui.dropdown(
        options={
            "rig 5-8 (dimmer)": "/Users/alex/Downloads/Ilmenau/openptv_illmenau_5678/parameters_Run1.yaml",
            "rig 1-4 (brighter)": "/Users/alex/Downloads/Ilmenau/openptv_illmenau_4cam/parameters_Run1.yaml",
        },
        value="rig 5-8 (dimmer)",
        label="rig (YAML)",
    )
    rig_select
    return (rig_select,)


@app.cell
def _(Path, rig_select):
    from openptv2.gui.parameter_manager import ParameterManager
    from openptv2.gui.ptv import py_start_proc_c, read_frame_images

    yaml_path = Path(rig_select.value)
    pm = ParameterManager()
    pm.from_yaml(yaml_path)
    cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(pm)
    num_cams = pm.num_cams
    first, last = int(spar.get_first()), int(spar.get_last())
    bub_defaults = dict(pm.parameters.get("bubble_detection", {}) or {})
    img_base_names = [
        str(yaml_path.parent / n) if not Path(n).is_absolute() else n
        for n in spar.img_base_name
    ]
    return (
        bub_defaults,
        first,
        img_base_names,
        last,
        num_cams,
        pm,
        read_frame_images,
        yaml_path,
    )


@app.cell
def _(first, last, mo):
    frame_ui = mo.ui.slider(
        first, last, step=1, value=min(10401, last),
        label="demo frame", show_value=True,
    )
    frame_ui
    return (frame_ui,)


@app.cell
def _(mo, num_cams):
    cam_ui = mo.ui.radio(
        options=[f"cam{c + 1}" for c in range(num_cams)],
        value="cam1",
        label="camera",
    )
    cam_ui
    return (cam_ui,)


@app.cell
def _(cam_ui):
    cam_idx = int(cam_ui.value.replace("cam", "")) - 1
    cam_idx
    return (cam_idx,)


@app.cell
def _(bub_defaults, mo):
    bg_on = mo.ui.checkbox(label="subtract static background", value=True)
    bg_frames_ui = mo.ui.slider(
        3, 51, step=2, value=int(bub_defaults.get("bg_frames", 11)),
        label="bg frames sampled", show_value=True,
    )
    bg_stride_ui = mo.ui.slider(
        1, 10, step=1, value=int(bub_defaults.get("bg_stride", 1)),
        label="bg stride", show_value=True,
    )
    bg_method_ui = mo.ui.dropdown(
        options=["median", "mean"],
        value=str(bub_defaults.get("bg_method", "median")),
        label="bg method",
    )
    bg_std_ui = mo.ui.slider(
        1.0, 10.0, step=0.5, value=float(bub_defaults.get("bg_std_factor", 3.0)),
        label="bg constant-pixel cutoff (x median std)", show_value=True,
    )
    bg_pol_ui = mo.ui.dropdown(
        options=["bright", "dark"],
        value=str(bub_defaults.get("bg_polarity", "bright")),
        label="bubble polarity",
    )
    mo.vstack([
        mo.md("### Static background (walls, seams, sensor pattern)"),
        mo.hstack([bg_on, bg_method_ui, bg_pol_ui]),
        mo.hstack([bg_frames_ui, bg_stride_ui, bg_std_ui]),
    ])
    return (
        bg_frames_ui,
        bg_method_ui,
        bg_on,
        bg_pol_ui,
        bg_std_ui,
        bg_stride_ui,
    )


@app.cell(hide_code=True)
def _(bub_defaults, mo):
    win_ui = mo.ui.slider(5, 51, step=2, value=int(bub_defaults.get("win", 25)),
                          label="local window (px)", show_value=True)
    z_ui = mo.ui.slider(3.0, 12.0, step=0.5, value=float(bub_defaults.get("z_thresh", 6.0)),
                        label="z threshold (local sigma)", show_value=True)
    merge_ui = mo.ui.slider(1, 10, step=1, value=int(bub_defaults.get("merge_radius", 4)),
                            label="glare merge radius (px)", show_value=True)
    nnmin_ui = mo.ui.slider(1, 20, step=1, value=int(bub_defaults.get("nnmin", 1)),
                            label="min pixels per blob", show_value=True)
    nnmax_ui = mo.ui.slider(100, 5000, step=100, value=int(bub_defaults.get("nnmax", 5000)),
                            label="max pixels per blob", show_value=True)
    x0_ui = mo.ui.slider(0, 2560, step=10, value=0, label="crop x0", show_value=True)
    y0_ui = mo.ui.slider(0, 2048, step=10, value=0, label="crop y0", show_value=True)
    crop_ui = mo.ui.slider(100, 2560, step=20, value=2560, label="crop size", show_value=True)
    mo.vstack([
        mo.md("### Detector (local z-score + glare merging)"),
        mo.hstack([win_ui, z_ui, merge_ui]),
        mo.hstack([nnmin_ui, nnmax_ui]),
        mo.hstack([x0_ui, y0_ui, crop_ui]),
    ])
    return crop_ui, merge_ui, nnmax_ui, nnmin_ui, win_ui, x0_ui, y0_ui, z_ui


@app.cell(hide_code=True)
def _(
    bg_frames_ui,
    bg_method_ui,
    bg_on,
    bg_pol_ui,
    bg_std_ui,
    bg_stride_ui,
    cam_idx,
    first,
    img_base_names,
    last,
    mo,
    np,
    num_cams,
    pm,
    read_frame_images,
):
    from openptv2.algorithms.bubble_background import (
        estimate_background,
        subtract_background,
    )

    _ = bg_frames_ui.value
    _ = bg_method_ui.value
    _ = bg_on.value
    _ = bg_pol_ui.value
    _ = bg_std_ui.value
    _ = bg_stride_ui.value
    _ = cam_idx

    K = int(bg_frames_ui.value)
    span = last - first
    sample = sorted(
        {min(first + round(i * span / max(K - 1, 1)), last) for i in range(K)}
    )
    st = int(bg_stride_ui.value)
    if st > 1 and len(sample) > 3:
        sample = [sample[0]] + sample[1::st]
        if sample[-1] != last:
            sample.append(last)

    stack = []
    for fr in sample:
        imgs = read_frame_images(pm, img_base_names, num_cams, int(fr))
        stack.append(np.asarray(imgs[cam_idx], dtype=np.uint8))
    bg_img, bg_info = estimate_background(
        np.stack(stack),
        method=bg_method_ui.value,
        std_factor=float(bg_std_ui.value),
        polarity=bg_pol_ui.value,
    )
    bg_status = mo.md(
        f"background from **{bg_info['K']}** frames "
        f"({sample[0]}..{sample[-1]}, {bg_info['method']}), "
        f"unstable pixels **{100 * bg_info['frac_unstable']:.2f}%** "
        f"(temporal min/max fallback), cutoff {bg_info['std_cutoff']:.2f}"
    )
    bg_status
    return bg_img, subtract_background


@app.cell(hide_code=True)
def _(
    bg_img,
    bg_on,
    bg_pol_ui,
    cam_idx,
    frame_ui,
    img_base_names,
    np,
    num_cams,
    plt,
    pm,
    read_frame_images,
    subtract_background,
):
    _ = frame_ui.value
    _ = cam_idx
    _ = bg_on.value
    _ = bg_pol_ui.value

    raw_list = read_frame_images(pm, img_base_names, num_cams, int(frame_ui.value))
    raw = np.asarray(raw_list[cam_idx], dtype=np.float64)
    if bool(bg_on.value):
        demo = np.clip(
            subtract_background(raw, bg_img, bg_pol_ui.value), 0.0, 255.0
        )
        demo_title = f"frame {int(frame_ui.value)} cam{cam_idx + 1} — background SUBTRACTED"
    else:
        demo = raw
        demo_title = f"frame {int(frame_ui.value)} cam{cam_idx + 1} — RAW (background kept)"

    fig0, axes0 = plt.subplots(1, 3, figsize=(15, 5))
    axes0[0].imshow(raw, cmap="gray", vmin=0, vmax=255)
    axes0[0].set_title("raw frame")
    axes0[0].axis("off")
    axes0[1].imshow(bg_img, cmap="gray")
    axes0[1].set_title("static background estimate")
    axes0[1].axis("off")
    axes0[2].imshow(demo, cmap="gray", vmin=0, vmax=255)
    axes0[2].set_title("demo input to detector")
    axes0[2].axis("off")
    fig0.suptitle(demo_title)
    fig0.tight_layout()
    fig0
    return demo, raw


@app.cell
def _(
    bg_on,
    crop_ui,
    demo,
    merge_ui,
    nnmax_ui,
    nnmin_ui,
    plt,
    raw,
    win_ui,
    x0_ui,
    y0_ui,
    z_ui,
):
    from openptv2.algorithms.bubble_detection import detect_bubbles_fast

    _ = bg_on.value
    _ = win_ui.value
    _ = z_ui.value
    _ = merge_ui.value
    _ = nnmin_ui.value
    _ = nnmax_ui.value

    kw = dict(win=int(win_ui.value), z_thresh=float(z_ui.value),
              merge_radius=int(merge_ui.value), nnmin=int(nnmin_ui.value),
              nnmax=int(nnmax_ui.value))
    t0x, t0y, t0n = detect_bubbles_fast(raw, **kw)
    t1x, t1y, t1n = detect_bubbles_fast(demo, **kw)

    x0, y0, cs = int(x0_ui.value), int(y0_ui.value), int(crop_ui.value)
    x1, y1 = min(x0 + cs, demo.shape[1]), min(y0 + cs, demo.shape[0])

    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 7))
    for ax, img, xs, ys, tag in [
        (axes1[0], raw, t0x, t0y, "WITHOUT background subtraction"),
        (axes1[1], demo, t1x, t1y, "WITH background subtraction"),
    ]:
        ax.imshow(img[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255)
        inside = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
        ax.scatter(xs[inside] - x0, ys[inside] - y0, s=18,
                   facecolors="none", edgecolors="red", linewidths=1.0)
        ax.set_title(f"{len(xs)} detections {tag}")
        ax.axis("off")
    fig1.tight_layout()
    fig1
    return


@app.cell
def _(mo):
    export_btn = mo.ui.run_button(label="write bubble_detection to this YAML")
    export_btn
    return (export_btn,)


@app.cell
def _(
    bg_frames_ui,
    bg_method_ui,
    bg_on,
    bg_pol_ui,
    bg_std_ui,
    bg_stride_ui,
    export_btn,
    merge_ui,
    mo,
    nnmax_ui,
    nnmin_ui,
    pm,
    win_ui,
    yaml_path,
    z_ui,
):
    if export_btn.value:
        import yaml

        block = {
            "win": int(win_ui.value),
            "z_thresh": float(z_ui.value),
            "merge_radius": int(merge_ui.value),
            "nnmin": int(nnmin_ui.value),
            "nnmax": int(nnmax_ui.value),
            "bg_subtract": bool(bg_on.value),
            "bg_frames": int(bg_frames_ui.value),
            "bg_stride": int(bg_stride_ui.value),
            "bg_method": str(bg_method_ui.value),
            "bg_std_factor": float(bg_std_ui.value),
            "bg_polarity": str(bg_pol_ui.value),
        }
        pm.parameters["bubble_detection"] = block
        with open(yaml_path) as fh:
            full = yaml.safe_load(fh)
        full["bubble_detection"] = block
        with open(yaml_path, "w") as fh:
            yaml.safe_dump(full, fh, sort_keys=False)
        result = mo.md(f"wrote `bubble_detection` to `{yaml_path}` (backup the file first!)")
    else:
        result = mo.md("*not saved — press the button to write the YAML*")
    result
    return


if __name__ == "__main__":
    app.run()
