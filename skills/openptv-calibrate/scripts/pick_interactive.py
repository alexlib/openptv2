# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.5.1",
#     "matplotlib",
#     "plotly==5.24.1",
#     "pyyaml==6.0.3",
#     "scikit-image",
#     "imageio==2.37.4",
# ]
# ///
"""Interactive (browser-based) manual-orientation point picker.

Unlike `calib.py pick` (a blocking desktop matplotlib window that walks you
through 4 fixed point IDs in a fixed order), this lets you click a point on
the camera image first, then set or correct which calibration-body ID that
click corresponds to -- comparing against the labeled 3D body map on the
right before adding it. Add as many points as you want per camera (4
minimum for a seed), switch cameras with the dropdown, then write.

Run with:
    uv run marimo edit --sandbox --no-token \\
        skills/openptv-calibrate/scripts/pick_interactive.py \\
        -- --target "<dataset-dir-or-yaml>"
"""

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.express as px

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    return Path, mo, np, plt, px, sys


@app.cell
def _(Path, mo):
    import yaml as _yaml

    from openptv2.autocalibration import _find_yaml, cam_files, resolve_calblock

    _target_arg = mo.cli_args().get("target")
    _target = Path(_target_arg).resolve() if _target_arg else mo.notebook_dir()
    base = _target.parent if (_target.is_file() and _target.suffix in (".yaml", ".yml")) else _target

    yaml_path = _find_yaml(base)
    if yaml_path is None:
        raise FileNotFoundError(f"no parameters_*.yaml found under {base}")
    cfg = _yaml.safe_load(yaml_path.read_text())
    num_cams = int(cfg.get("num_cams") or cfg["ptv"].get("num_cams"))
    ptv_params = cfg["ptv"]
    calblock_path = resolve_calblock(base)
    return base, cam_files, calblock_path, num_cams, ptv_params


@app.cell
def _(calblock_path, np):
    def _read_calblock(path):
        ids, pts = [], []
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 4:
                ids.append(int(float(parts[0])))
                pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return np.array(ids), np.array(pts)

    body_ids, body_xyz = _read_calblock(calblock_path)
    return body_ids, body_xyz


@app.cell
def _(base, cam_files, num_cams, ptv_params):
    # Resolve each camera's own view. Splitter datasets share one raw frame
    # (image_split turns it into per-camera views in memory); otherwise each
    # camera has its own image file.
    import imageio.v3 as iio

    if ptv_params.get("splitter"):
        from openptv2.gui.ptv import image_split

        img0_path, _, _ = cam_files(base, 0)
        raw = iio.imread(img0_path)
        if raw.ndim > 2:
            from skimage.color import rgb2gray
            from skimage.util import img_as_ubyte

            raw = img_as_ubyte(rgb2gray(raw[:, :, :3]))
        views = image_split(raw, order=ptv_params.get("splitter_order") or [0, 1, 3, 2])
    else:
        views = []
        for _cam in range(num_cams):
            _img_path, _, _ = cam_files(base, _cam)
            views.append(iio.imread(_img_path))
    return (views,)


@app.cell
def _(mo, num_cams):
    cam_select = mo.ui.dropdown(
        options=[str(i + 1) for i in range(num_cams)], value="1", label="Camera"
    )
    id_input = mo.ui.number(start=1, stop=100_000, value=1, label="Calibration-body point ID")
    mo.hstack([cam_select, id_input])
    return cam_select, id_input


@app.cell
def _(mo):
    # {"1": [{"id": 1, "x": .., "y": ..}, ...], "2": [...], ...}
    get_points, set_points = mo.state({})
    return get_points, set_points


@app.cell
def _(cam_select, mo, px, views):
    _img = views[int(cam_select.value) - 1]
    _fig = px.imshow(_img, color_continuous_scale="gray")
    _fig.update_layout(
        title=f"cam{cam_select.value}: click a point on the image",
        dragmode=False,
        clickmode="event+select",
        height=560,
    )
    click_plot = mo.ui.plotly(_fig)
    click_plot
    return (click_plot,)


@app.cell
def _(cam_select, click_plot, get_points, id_input, mo, set_points):
    def _add_current_click(_v):
        pts_now = click_plot.points
        if not pts_now:
            return _v
        p = pts_now[-1]
        cam = cam_select.value
        pid = int(id_input.value)
        current = dict(get_points())
        lst = list(current.get(cam, []))
        lst.append({"id": pid, "x": float(p["x"]), "y": float(p["y"])})
        current[cam] = lst
        set_points(current)
        return _v + 1

    add_button = mo.ui.button(label="Add point with this ID", value=0, on_click=_add_current_click)

    def _clear_camera(_v):
        current = dict(get_points())
        current[cam_select.value] = []
        set_points(current)
        return _v + 1

    clear_button = mo.ui.button(label="Clear this camera's points", value=0, on_click=_clear_camera)

    mo.hstack([add_button, clear_button])
    return add_button, clear_button


@app.cell
def _(click_plot, mo):
    _pts = click_plot.points
    _msg = f"last click: x={_pts[-1]['x']:.1f}, y={_pts[-1]['y']:.1f}" if _pts else "no click yet -- click the image above"
    mo.md(_msg)
    return


@app.cell
def _(body_ids, body_xyz, id_input, mo, plt):
    _fig, _ax = plt.subplots(figsize=(6, 5))
    _ax.scatter(body_xyz[:, 0], body_xyz[:, 1], s=25, c="lightgray")
    for _pid, _p in zip(body_ids, body_xyz):
        _ax.annotate(str(_pid), (_p[0], _p[1]), fontsize=6,
                     textcoords="offset points", xytext=(2, 2))
    if int(id_input.value) in set(body_ids.tolist()):
        _hp = body_xyz[list(body_ids).index(int(id_input.value))]
        _ax.scatter([_hp[0]], [_hp[1]], s=220, facecolors="none",
                    edgecolors="red", linewidths=2.0)
        _ax.set_title(f"3D body -- ID {int(id_input.value)} circled in red")
    else:
        _ax.set_title(f"ID {int(id_input.value)} is not in this calibration body!", color="red")
    _ax.set_xlabel("X (left -> right)")
    _ax.set_ylabel("Y (bottom -> top)")
    _ax.set_aspect("equal")
    plt.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _(cam_select, get_points, mo):
    _pts = get_points().get(cam_select.value, [])
    if _pts:
        points_table = mo.ui.table(
            data=[{"#": i, "id": p["id"], "x": round(p["x"], 1), "y": round(p["y"], 1)}
                  for i, p in enumerate(_pts)],
            label=f"cam{cam_select.value} points",
        )
    else:
        points_table = mo.md(f"cam{cam_select.value}: no points added yet.")
    points_table
    return


@app.cell
def _(get_points, mo, num_cams):
    _summary = "\n".join(
        f"- cam{c + 1}: {len(get_points().get(str(c + 1), []))} points"
        for c in range(num_cams)
    )
    mo.md(f"**Progress**\n\n{_summary}")
    return


@app.cell
def _(base, get_points, mo, num_cams):
    def _write(_v):
        pts = get_points()
        ready = all(len(pts.get(str(c + 1), [])) == 4 for c in range(num_cams))
        if not ready:
            return _v
        import calib  # skills/openptv-calibrate/scripts/calib.py, same directory

        seeds = {
            str(c): [[p["id"], p["x"], p["y"]] for p in pts[str(c + 1)]]
            for c in range(num_cams)
        }
        calib._write_seed(base, seeds, num_cams)
        return _v + 1

    write_button = mo.ui.button(
        label="Write seed (man_ori) to YAML + .par/.dat", value=0, on_click=_write
    )
    _ready = all(len(get_points().get(str(c + 1), [])) == 4 for c in range(num_cams))
    mo.vstack([
        mo.md("Every camera needs exactly 4 points before this writes anything."
              if not _ready else "Ready to write."),
        write_button,
        mo.md(f"Wrote man_ori seed for {num_cams} cameras to `{base}`."
              if write_button.value else ""),
    ])
    return


if __name__ == "__main__":
    app.run()
