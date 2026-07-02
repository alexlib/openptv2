# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.20.0",
#     "numpy>=2.0.0",
#     "plotly>=5.0.0",
#     "openptv2",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    return go, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # OpenPTV2 Visual Hull Viewer

    Samples a 3D grid around the camera convergence point and identifies
    voxels visible in all camera views. Requires calibration `.ori` + `.addpar`
    files to be loaded.
    """)
    return


@app.cell
def _(mo):
    yaml_input = mo.ui.text(
        label="YAML parameter file",
        value="test_data/test_cavity/parameters_Run1.yaml",
        full_width=True,
    )
    side_slider = mo.ui.slider(10, 500, value=100, step=10, label="Box half-size (mm)")
    res_slider = mo.ui.slider(5, 40, value=15, step=5, label="Voxel resolution")
    load_btn = mo.ui.run_button(label="Compute")

    mo.vstack([yaml_input, mo.hstack([side_slider, res_slider]), load_btn])
    return load_btn, res_slider, side_slider, yaml_input


@app.cell
def _(load_btn, mo, np, yaml_input):
    from pathlib import Path

    if not load_btn.value:
        mo.stop("Click Compute to run.")

    from openptv2.gui.pyptv.parameter_manager import ParameterManager
    from openptv2.gui.pyptv.ptv import py_start_proc_c
    from openptv2 import image_coordinates

    yaml_path = Path(yaml_input.value).expanduser().resolve()
    if not yaml_path.exists():
        mo.stop(f"File not found: {yaml_path}")

    pm = ParameterManager(yaml_path)
    cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(pm)
    num_cams = cpar.get_num_cams()

    print(f"Loaded {num_cams} cameras from {yaml_path}")
    return cals, cpar, image_coordinates, num_cams


@app.cell
def _(cals, cpar, go, mo, np, num_cams, res_slider, side_slider):
    def _find_convergence(cals):
        """Least-squares convergence point of all optical axes."""
        origins, dirs = [], []
        for cal in cals:
            ext = cal.ext_par
            R = np.array(cal.get_rotation_matrix()) if hasattr(cal, "get_rotation_matrix") else np.eye(3)
            C = np.array([ext.x0, ext.y0, ext.z0])
            D = R.T @ np.array([0.0, 0.0, 1.0])
            origins.append(C)
            dirs.append(D / np.linalg.norm(D))

        S = np.zeros((3, 3))
        b = np.zeros(3)
        for O, D in zip(origins, dirs):
            M = np.eye(3) - np.outer(D, D)
            S += M
            b += M @ O
        return np.linalg.solve(S, b)

    center = _find_convergence(cals)

    half = side_slider.value
    res = res_slider.value
    ax_pts = np.linspace(-half, half, res)
    X, Y, Z = np.meshgrid(ax_pts, ax_pts, ax_pts)
    pts = np.column_stack([X.ravel() + center[0],
                           Y.ravel() + center[1],
                           Z.ravel() + center[2]])

    imx, imy = cpar.get_image_size()
    mask = np.ones(len(pts), dtype=bool)

    for cal in cals:
        # project each point and check sensor bounds
        for i, pt in enumerate(pts):
            if not mask[i]:
                continue
            try:
                px, py = cal.flat_image_coord(pt[0], pt[1], pt[2])
                if not (0 <= px <= imx and 0 <= py <= imy):
                    mask[i] = False
            except Exception:
                mask[i] = False

    visible = pts[mask]
    print(f"Convergence center: {center.round(1)}, visible voxels: {mask.sum()}/{len(pts)}")

    fig = go.Figure(data=go.Scatter3d(
        x=visible[:, 0], y=visible[:, 1], z=visible[:, 2],
        mode="markers",
        marker=dict(size=3, color=visible[:, 2], colorscale="Viridis", opacity=0.6),
    ))
    fig.update_layout(
        title=f"Visual hull — {mask.sum()} voxels visible from all {num_cams} cameras",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    mo.ui.plotly(fig)
    return


if __name__ == "__main__":
    app.run()
