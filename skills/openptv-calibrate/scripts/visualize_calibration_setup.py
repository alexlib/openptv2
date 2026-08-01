# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.5.1",
#     "matplotlib",
#     "imageio==2.37.4",
#     "pyyaml==6.0.3",
# ]
# ///
"""Interactive 3D view of the calibrated multi-camera setup.

Runs directly from this skills/ checkout -- no per-dataset copy needed. Pass
the dataset via --dataset after `--` (marimo's CLI-args convention); falls
back to `mo.notebook_dir()` if omitted, so opening a copy still works too,
but that is no longer the normal way to use this. Open with the marimo-pair
skill:
    uv run marimo edit --sandbox --no-token \\
        skills/openptv-calibrate/scripts/visualize_calibration_setup.py \\
        -- --dataset "<dataset>"

Plots the calibration body (calblock points, each labeled with its point ID),
the world coordinate frame at the origin, each camera's position +
orientation (from its .ori file, labeled with its splitter quadrant from
ptv.splitter_order), and a 2x2 grid of per-camera detected-vs-reprojected
overlays (also ID-labeled, read from cal/calib_matches/ -- run
dump_matches.py first to generate those). Drag directly on the 3D plot with
the mouse to rotate (uses mo.mpl.interactive, backed by matplotlib's real
WebAgg backend -- not slider-driven).
"""

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import imageio.v3 as iio
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return Path, iio, mo, np, plt


@app.cell
def _(Path, mo):
    # --dataset (after `--`) is the normal way to point this at a dataset;
    # mo.notebook_dir() is only a fallback for a locally-copied notebook.
    _dataset_arg = mo.cli_args().get("dataset")
    dataset_dir = Path(_dataset_arg).resolve() if _dataset_arg else mo.notebook_dir()
    cal_dir = dataset_dir / "cal"
    return cal_dir, dataset_dir


@app.cell
def _(np):
    def read_ori(path):
        """Parse a classic OpenPTV .ori file -> (position[3], rotation_matrix[3,3])."""
        vals = [float(v) for v in path.read_text().split()]
        pos = np.array(vals[0:3])
        dm = np.array(vals[6:15]).reshape(3, 3)
        return pos, dm

    def read_calblock(path):
        """Parse 'id x y z' lines -> (n, 3) array, skipping blank/short lines."""
        pts = []
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 4:
                pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return np.array(pts)

    return read_calblock, read_ori


@app.cell
def _(dataset_dir, cal_dir):
    # Camera image/.ori paths and the calblock path come from the dataset
    # YAML's cal_ori: block when present -- the same files the GUI reads and
    # writes, not a separate camN.tif naming convention that can silently
    # drift out of sync with the real dataset (as happened once already:
    # adapter copies under a camN.tif convention got cleaned up as apparent
    # clutter, breaking every tool that assumed they'd always be there).
    import yaml as _yaml_paths

    _yaml_path = dataset_dir / "parameters_Run1.yaml"
    _cfg = _yaml_paths.safe_load(_yaml_path.read_text()) if _yaml_path.exists() else {}
    _cal_ori = _cfg.get("cal_ori") or {}
    _img_cal_name = _cal_ori.get("img_cal_name") or []
    _img_ori = _cal_ori.get("img_ori") or []
    _fixp_name = _cal_ori.get("fixp_name")

    if _img_cal_name and _img_ori:
        cam_paths = [
            (f"cam{i + 1}", dataset_dir / _img_cal_name[i], dataset_dir / _img_ori[i])
            for i in range(len(_img_cal_name))
        ]
    else:
        cam_paths = [
            (p.stem.split(".")[0], cal_dir / (p.stem.split(".")[0] + ".tif"), p)
            for p in sorted(cal_dir.glob("cam[0-9].tif.ori"))
        ]

    calblock_path = (
        (dataset_dir / _fixp_name) if _fixp_name else (cal_dir / "target_on_a_side.txt")
    )
    return cam_paths, calblock_path


@app.cell
def _(calblock_path, cam_paths, read_calblock, read_ori):
    body = read_calblock(calblock_path)
    cams = [(name, *read_ori(ori)) for name, _, ori in cam_paths]
    return body, cams


@app.cell
def _(body, cam_quadrant, cams, mo, np, plt):
    def plot_axes(ax, origin, dm, length, label, flip_z=False):
        """Draw an RGB=XYZ triad. For camera frames, dm columns are this .ori
        format's local axes in world coordinates, but column 2 (Z) points
        *backward* out of the lens (denom = dm[:,2] . (world - pos) is
        negative for points the camera sees -- verify against
        openptv2.algorithms.imgcoord.img_coord on your own dataset before
        trusting this blindly) -- so flip_z=True shows the actual viewing
        direction. The world frame is drawn as-is (flip_z=False)."""
        z = -dm[:, 2] if flip_z else dm[:, 2]
        axes = (dm[:, 0], dm[:, 1], z)
        colors = ("r", "g", "b")
        for vec, c in zip(axes, colors):
            ax.quiver(
                *origin, *(vec * length), color=c, linewidth=2, arrow_length_ratio=0.15
            )
        ax.text(*origin, "  " + label, fontsize=9, weight="bold")

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        body[:, 0],
        body[:, 1],
        body[:, 2],
        s=8,
        c="gray",
        alpha=0.6,
        label="calibration body",
    )
    for _pid, (_bx, _by, _bz) in enumerate(body, start=1):
        ax.text(_bx, _by, _bz, str(_pid), fontsize=4, color="dimgray")

    span = float(np.ptp(np.vstack([body] + [pos for _, pos, _ in cams]), axis=0).max())
    axis_len = span * 0.12

    plot_axes(ax, np.zeros(3), np.eye(3), axis_len * 1.5, "world")
    for name, pos, dm in cams:
        cam_label = (
            name + " (" + cam_quadrant[name] + ")" if name in cam_quadrant else name
        )
        ax.scatter(*pos, s=60, c="k", marker="^")
        plot_axes(ax, pos, dm, axis_len, cam_label, flip_z=True)

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    title_line1 = "Calibration setup: world frame, calibration body, camera poses"
    title_line2 = "(R/G/B arrows = local X/Y/Z axes; gray numbers = calibration-body point ID; drag to rotate)"
    ax.set_title(title_line1 + chr(10) + title_line2)
    ax.legend(loc="upper left")
    ax.view_init(elev=20, azim=-35)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass
    plt.tight_layout()
    mo.mpl.interactive(fig)
    return


@app.cell
def _(cam_paths, np, read_ori):
    def test_ori_files_are_orthonormal():
        """Smoke test: parsed rotation matrices must be orthonormal (det ~= +-1)."""
        for _, _, ori in cam_paths:
            _, dm = read_ori(ori)
            det = np.linalg.det(dm)
            assert abs(abs(det) - 1.0) < 1e-3, (
                f"{ori.name}: not a rotation matrix (det={det})"
            )
            assert np.allclose(dm @ dm.T, np.eye(3), atol=1e-3), (
                f"{ori.name}: not orthonormal"
            )

    return


@app.cell
def _(cal_dir, np):
    def read_matches(path):
        """Parse 'id det_x det_y rep_x rep_y' lines -> dict of (n,) / (n,2) arrays."""
        ids, det, rep = [], [], []
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            ids.append(int(parts[0]))
            det.append((float(parts[1]), float(parts[2])))
            rep.append((float(parts[3]), float(parts[4])))
        return np.array(ids), np.array(det), np.array(rep)

    match_files = sorted(cal_dir.glob("calib_matches/cam[0-9]_matches.txt"))
    cam_matches = {p.stem.split("_")[0]: read_matches(p) for p in match_files}
    return (cam_matches,)


@app.cell
def _(cam_paths, cam_matches, iio, np, plt):
    overlay_fig, overlay_axes = plt.subplots(2, 2, figsize=(14, 12))
    _img_by_name = {name: img for name, img, _ in cam_paths}

    for _ax, (_cam_name, (_ids, _det, _rep)) in zip(
        overlay_axes.flat, sorted(cam_matches.items())
    ):
        _img = iio.imread(_img_by_name[_cam_name])
        _ax.imshow(_img, cmap="gray")
        _ax.scatter(
            _det[:, 0],
            _det[:, 1],
            s=40,
            facecolors="none",
            edgecolors="lime",
            linewidths=1.2,
            label="detected",
        )
        _ax.scatter(_rep[:, 0], _rep[:, 1], s=8, c="red", label="reprojected")
        for _pid, (_x, _y) in zip(_ids, _det):
            _ax.annotate(
                str(_pid),
                (_x, _y),
                fontsize=6,
                color="yellow",
                textcoords="offset points",
                xytext=(3, 3),
            )
        _rms = float(np.sqrt(np.mean(np.sum((_det - _rep) ** 2, axis=1))))
        _ax.set_title(
            f"{_cam_name}  RMS={_rms:.3f}px  n={len(_ids)}  (yellow = calibration-body point ID)"
        )
        _ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

    plt.tight_layout()
    overlay_fig
    return


@app.cell
def _(dataset_dir):
    import yaml as _yaml

    QUADRANT_NAMES = ["top-left", "top-right", "bottom-left", "bottom-right"]

    _yaml_path = dataset_dir / "parameters_Run1.yaml"
    _cfg = _yaml.safe_load(_yaml_path.read_text()) if _yaml_path.exists() else {}
    splitter_order = (_cfg.get("ptv", {}) or {}).get("splitter_order") or [0, 1, 3, 2]

    cam_quadrant = {
        f"cam{i + 1}": QUADRANT_NAMES[splitter_order[i]]
        for i in range(len(splitter_order))
    }
    return (cam_quadrant,)


if __name__ == "__main__":
    app.run()
