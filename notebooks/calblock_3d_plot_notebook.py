import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full", app_title="3D Calibration Target Visualizer - Marimo Notebook")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    return mo, np, go


@app.cell
def _(mo):
    mo.md(
        """
        # 🎯 3D Calibration Target Visualizer
        Interactive visualization and spatial analysis of calibration targets (OpenPTV format).
        """
    )
    return


@app.cell
def _(mo):
    cal_file_input = mo.ui.text(
        value=r"C:\Users\alex\Downloads\hidimaging_test\LV\calibration\cal\calblock_90deg_clockwise.txt",
        label="Calibration File Path",
        full_width=True,
    )

    show_surface = mo.ui.checkbox(value=True, label="Show Surface Mesh")
    opacity_slider = mo.ui.slider(start=0.0, stop=1.0, step=0.05, value=0.5, label="Surface Opacity")
    marker_size_slider = mo.ui.slider(start=1, stop=15, step=1, value=6, label="Marker Size")
    colorscale_dropdown = mo.ui.dropdown(
        options=["Plasma", "Viridis", "Cividis", "Magma", "Turbo", "Rainbow"],
        value="Plasma",
        label="Color Palette",
    )

    return (
        cal_file_input,
        colorscale_dropdown,
        marker_size_slider,
        opacity_slider,
        show_surface,
    )


@app.cell
def _(cal_file_input, colorscale_dropdown, marker_size_slider, mo, opacity_slider, show_surface):
    mo.vstack([
        cal_file_input,
        mo.hstack([
            show_surface,
            opacity_slider,
            marker_size_slider,
            colorscale_dropdown,
        ], justify="start", align="center", gap=2)
    ])
    return


@app.cell
def _(cal_file_input, np):
    file_path = cal_file_input.value
    data = []
    load_error = None
    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    pid = int(parts[0])
                    x_val, y_val, z_val = float(parts[1]), float(parts[2]), float(parts[3])
                    data.append((pid, x_val, y_val, z_val))
        data = np.array(data)
    except Exception as e:
        data = np.array([])
        load_error = str(e)
    return data, file_path, load_error


@app.cell
def _(data, load_error, mo, np):
    if load_error or len(data) == 0:
        stats_view = mo.md(f"⚠️ **Error loading file:** {load_error or 'No data found'}")
        pids, xs, ys, zs, u_x, u_y, u_z = [], [], [], [], [], [], []
    else:
        pids = data[:, 0].astype(int)
        xs = data[:, 1]
        ys = data[:, 2]
        zs = data[:, 3]

        num_pts = len(pids)
        u_x = np.unique(xs)
        u_y = np.unique(ys)
        u_z = np.unique(zs)

        stats_view = mo.md(
            f"""
            ### 📊 Dataset Summary
            - **Total Calibration Points:** `{num_pts}`
            - **Grid Topology:** `{len(u_y)} rows × {len(u_x)} columns`
            - **X Bounds:** `{xs.min():.1f}` to `{xs.max():.1f}` mm
            - **Y Bounds:** `{ys.min():.1f}` to `{ys.max():.1f}` mm
            - **Z Depth Bounds:** `{zs.min():.1f}` to `{zs.max():.1f}` mm
            """
        )
    return pids, stats_view, u_x, u_y, u_z, xs, ys, zs


@app.cell
def _(stats_view):
    stats_view
    return


@app.cell
def _(colorscale_dropdown, data, go, marker_size_slider, mo, np, opacity_slider, pids, show_surface, u_x, u_y, xs, ys, zs):
    if len(data) == 0:
        plot_view = mo.md("No plot to render.")
    else:
        fig = go.Figure()

        num_rows = len(u_y)
        num_cols = len(u_x)

        if num_rows * num_cols == len(data):
            grid_x = xs.reshape((num_rows, num_cols))
            grid_y = ys.reshape((num_rows, num_cols))
            grid_z = zs.reshape((num_rows, num_cols))

            if show_surface.value:
                fig.add_trace(go.Surface(
                    x=grid_x,
                    y=grid_y,
                    z=grid_z,
                    colorscale=colorscale_dropdown.value,
                    opacity=opacity_slider.value,
                    showscale=True,
                    colorbar=dict(title='Z (mm)', len=0.75),
                    hoverinfo='none',
                    name='Surface'
                ))

            # Grid wireframe lines
            for r in range(num_rows):
                fig.add_trace(go.Scatter3d(
                    x=grid_x[r, :],
                    y=grid_y[r, :],
                    z=grid_z[r, :],
                    mode='lines',
                    line=dict(color='rgba(180, 180, 180, 0.4)', width=2),
                    showlegend=False,
                    hoverinfo='none'
                ))
            for c in range(num_cols):
                fig.add_trace(go.Scatter3d(
                    x=grid_x[:, c],
                    y=grid_y[:, c],
                    z=grid_z[:, c],
                    mode='lines',
                    line=dict(color='rgba(180, 180, 180, 0.4)', width=2),
                    showlegend=False,
                    hoverinfo='none'
                ))

        hover_text = [
            f"<b>Point ID:</b> {pid}<br><b>X:</b> {px:.1f} mm<br><b>Y:</b> {py:.1f} mm<br><b>Z:</b> {pz:.1f} mm"
            for pid, px, py, pz in zip(pids, xs, ys, zs)
        ]

        fig.add_trace(go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode='markers+text',
            text=[str(pid) for pid in pids],
            textposition='top center',
            textfont=dict(size=8, color='white'),
            marker=dict(
                size=marker_size_slider.value,
                color=zs,
                colorscale=colorscale_dropdown.value,
                showscale=not show_surface.value,
                line=dict(color='white', width=1)
            ),
            hovertext=hover_text,
            hoverinfo='text',
            name='Target Markers'
        ))

        fig.update_layout(
            title=dict(
                text="<b>3D Calibration Target Plot</b>",
                font=dict(size=18, color='white'),
                x=0.5
            ),
            template="plotly_dark",
            scene=dict(
                xaxis=dict(
                    title="X (mm)",
                    backgroundcolor="rgb(20, 24, 33)",
                    gridcolor="rgb(60, 65, 80)",
                ),
                yaxis=dict(
                    title="Y (mm)",
                    backgroundcolor="rgb(20, 24, 33)",
                    gridcolor="rgb(60, 65, 80)",
                ),
                zaxis=dict(
                    title="Z (mm)",
                    backgroundcolor="rgb(20, 24, 33)",
                    gridcolor="rgb(60, 65, 80)",
                ),
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=-1.5, z=1.2)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            paper_bgcolor="rgb(15, 17, 23)",
            plot_bgcolor="rgb(15, 17, 23)",
            height=700
        )

        plot_view = mo.ui.plotly(fig)

    return fig, plot_view


@app.cell
def _(plot_view):
    plot_view
    return


if __name__ == "__main__":
    app.run()
