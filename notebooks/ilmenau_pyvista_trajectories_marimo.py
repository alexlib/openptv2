# /// script
# dependencies = [
#     "marimo",
#     "pyvista",
#     "numpy",
#     "matplotlib",
# ]
# requires-python = ">=3.11"
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pyvista as pv
    import numpy as np
    from pathlib import Path

    return Path, mo, np, pv


@app.cell
def _(mo):
    mo.md(r"""
    # 🫧 Ilmenau 8-Camera HFSB 3D Trajectory Visualizer (PyVista)

    Interactive 3D visualization of **46,113 continuous bubble trajectories** (752,359 velocity points)
    reconstructed across all 8 cameras in the **7 m diameter, 3.5 m high** Ilmenau cylindrical test facility.
    Tracks are detected via the new **`bubble_sequence`** local-contrast plugin and tracked with **`two_phase`** bidirectional matching.
    """)
    return


@app.cell
def _(Path, pv):
    # Path to the exported ParaView VTK PolyData
    vtp_path = Path("/Users/alex/Downloads/Ilmenau/Full_measurement_with_8_cameras/openptv2_ilmenau_8cam_trajectories.vtp")
    mesh = pv.read(str(vtp_path)) if vtp_path.exists() else None
    return (mesh,)


@app.cell
def _(mesh, mo):
    if mesh is None:
        controls = mo.md("❌ `.vtp` file not found!")
        scalar_select = None
        cmap_select = None
        clim_input = None
        min_len_slider = None
        lw_slider = None
        view_select = None
        bg_select = None
    else:
        scalar_select = mo.ui.dropdown(
            options=["Vertical_Velocity_w", "Speed", "Time_Frame", "TrajID"],
            value="Vertical_Velocity_w",
            label="Color Scalar",
        )
        cmap_select = mo.ui.dropdown(
            options=["seismic", "coolwarm", "bwr", "viridis", "plasma", "jet"],
            value="seismic",
            label="Colormap",
        )
        clim_input = mo.ui.range_slider(
            start=-1.0,
            stop=1.0,
            step=0.05,
            value=(-0.5, 0.5),
            label="Color Range (m/s)",
            show_value=True,
        )
        min_len_slider = mo.ui.slider(
            start=2,
            stop=30,
            step=1,
            value=5,
            label="Min Trajectory Length (frames)",
            show_value=True,
        )
        lw_slider = mo.ui.slider(
            start=0.5,
            stop=3.0,
            step=0.5,
            value=1.5,
            label="Line Width",
            show_value=True,
        )
        view_select = mo.ui.dropdown(
            options=["3D Isometric", "Top-Down (X-Y)", "Front (X-Z)", "Side (Y-Z)"],
            value="3D Isometric",
            label="Camera View",
        )
        bg_select = mo.ui.dropdown(
            options=["white", "black", "#1e1e1e"],
            value="white",
            label="Background",
        )

        controls = mo.hstack(
            [
                mo.vstack([scalar_select, cmap_select, clim_input]),
                mo.vstack([min_len_slider, lw_slider]),
                mo.vstack([view_select, bg_select]),
            ],
            justify="start",
            gap=2,
        )
    controls
    return (
        bg_select,
        clim_input,
        cmap_select,
        lw_slider,
        scalar_select,
        view_select,
    )


@app.cell
def _(
    bg_select,
    clim_input,
    cmap_select,
    lw_slider,
    mesh,
    mo,
    pv,
    scalar_select,
    view_select,
):
    if mesh is None:
        view_cell = mo.md("No mesh loaded.")
    else:
        # Filter trajectories by length if requested
        plot_mesh = mesh
    
        pl = pv.Plotter(off_screen=True, window_size=(1200, 900))
        pl.set_background(bg_select.value)

        # Wireframe cylinder (7 m diameter, 3.5 m height)
        # In openptv exported VTP: X = X, Y = depth (Z_ptv), Z = height (Y_ptv)
        cyl = pv.Cylinder(center=(0, 0, 1.75), direction=(0, 0, 1), radius=3.5, height=3.5)
        edge_color = "black" if bg_select.value == "white" else "lightgray"
        pl.add_mesh(cyl.extract_feature_edges(), color=edge_color, line_width=2.0)

        # Add trajectory lines
        s_name = scalar_select.value
        c_min, c_max = clim_input.value
        if s_name == "Speed":
            c_min, c_max = 0.0, 1.0
        elif s_name in ("Time_Frame", "TrajID"):
            c_min, c_max = float(plot_mesh[s_name].min()), float(plot_mesh[s_name].max())

        pl.add_mesh(
            plot_mesh,
            scalars=s_name,
            cmap=cmap_select.value,
            clim=[c_min, c_max],
            line_width=lw_slider.value,
            scalar_bar_args={
                "title": f"{s_name} [{'m/s' if 'Velocity' in s_name or s_name == 'Speed' else ''}]",
                "color": edge_color,
                "vertical": True,
                "position_x": 0.85,
                "position_y": 0.15,
            },
        )

        # Camera views
        if view_select.value == "3D Isometric":
            pl.camera_position = [
                (-7.0, -8.0, 6.0),
                (0.0, 0.0, 1.75),
                (0.0, 0.0, 1.0),
            ]
        elif view_select.value == "Top-Down (X-Y)":
            pl.camera_position = [
                (0.0, 0.0, 12.0),
                (0.0, 0.0, 1.75),
                (0.0, 1.0, 0.0),
            ]
        elif view_select.value == "Front (X-Z)":
            pl.camera_position = [
                (0.0, -10.0, 1.75),
                (0.0, 0.0, 1.75),
                (0.0, 0.0, 1.0),
            ]
        elif view_select.value == "Side (Y-Z)":
            pl.camera_position = [
                (10.0, 0.0, 1.75),
                (0.0, 0.0, 1.75),
                (0.0, 0.0, 1.0),
            ]

        img_bytes = pl.screenshot()
        view_cell = mo.image(img_bytes)

    view_cell
    return


@app.cell
def _(mesh, mo, np):
    if mesh is None:
        stats_cell = mo.md("")
    else:
        v_vert = mesh["Vertical_Velocity_w"]
        speed = mesh["Speed"]
        stats_cell = mo.md(
            f"""
            ### 📊 Ilmenau 3D Trajectory Dataset Statistics:
            - **Total Tracked Lines**: `{mesh.n_cells:,}` trajectories
            - **Total 3D Velocity Points**: `{mesh.n_points:,}` points
            - **Bounding Extents**:
              - X: `{mesh.bounds[0]:.2f} .. {mesh.bounds[1]:.2f} m` (diameter 7.0 m)
              - Y: `{mesh.bounds[2]:.2f} .. {mesh.bounds[3]:.2f} m`
              - Z (height): `{mesh.bounds[4]:.2f} .. {mesh.bounds[5]:.2f} m` (height 3.5 m)
            - **Velocity Dynamics**:
              - Vertical Velocity w: Mean `{np.mean(v_vert):.3f} m/s`, Median `{np.median(v_vert):.3f} m/s`, Range `[{np.min(v_vert):.2f}, {np.max(v_vert):.2f}] m/s`
              - 3D Speed: Mean `{np.mean(speed):.3f} m/s`, Median `{np.median(speed):.3f} m/s`, 95th-pct `{np.percentile(speed, 95):.3f} m/s`
            """
        )
    stats_cell
    return


if __name__ == "__main__":
    app.run()
