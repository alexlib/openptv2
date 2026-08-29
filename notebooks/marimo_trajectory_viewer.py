import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import zarr

    return go, mo, np, zarr


@app.cell
def _(mo):
    mo.md("# Trajectory Viewer — TT13 aorta WP1")
    return


@app.cell
def _(mo):
    zarr_path = mo.ui.text(value=r"C:\Users\alex\Downloads\TT13_aorta\wp1\res\run.zarr", label="Zarr store")
    ui_n = mo.ui.slider(start=10, stop=500, step=10, value=100, label="Show top N longest")
    ui_color = mo.ui.dropdown(options=["Speed", "Frame"], value="Speed", label="Color by")
    mo.vstack([zarr_path, mo.hstack([ui_n, ui_color], gap=2)])
    return ui_color, ui_n, zarr_path


@app.cell
def _(mo, np, ui_n, zarr_path):
    root = zarr.open_group(zarr_path.value, mode="r")

    traj_idx = root["traj"]
    idx_trajid = np.asarray(traj_idx["trajid"])
    idx_length = np.asarray(traj_idx["length"])
    idx_first_row = np.asarray(traj_idx["first_row"])

    order = np.argsort(idx_length)[::-1][: ui_n.value]
    top_trajids = idx_trajid[order]
    top_lengths = idx_length[order]
    top_first_row = idx_first_row[order]

    mo.md(f"Top {ui_n.value} longest: **{top_lengths[0]}** to **{top_lengths[-1]}** frames "
          f"(out of {len(idx_trajid)} total)")
    return idx_length, root, top_first_row, top_lengths, top_trajids


@app.cell
def _(go, idx_length, np):
    _counts, _bins = np.histogram(idx_length[idx_length > 0], bins=100)
    fig_hist = go.Figure(go.Bar(x=_bins[:-1], y=_counts, width=np.diff(_bins),
                                marker_color="steelblue"))
    fig_hist.update_layout(
        xaxis_title="Trajectory length [frames]", yaxis_title="Count",
        title=f"Length distribution ({len(idx_length)} trajectories, "
              f"median={np.median(idx_length):.0f}, mean={np.mean(idx_length):.1f})",
        width=900, height=350, margin=dict(l=50, r=20, t=40, b=50),
    )
    fig_hist.add_vline(x=10, line_dash="dash", line_color="red", annotation_text="min=10")
    mo.ui.plotly(fig_hist)


@app.cell
def _(np, root, top_first_row, top_lengths, top_trajids):
    pos_arr = root["trajectories"]["pos"]
    time_arr = root["trajectories"]["time"]

    traj_pos = []
    traj_time = []
    for _i in range(len(top_trajids)):
        _lo = int(top_first_row[_i])
        _hi = _lo + int(top_lengths[_i])
        traj_pos.append(np.asarray(pos_arr[_lo:_hi]))
        traj_time.append(np.asarray(time_arr[_lo:_hi]))

    total_pts = sum(len(p) for p in traj_pos)
    mo.md(f"Loaded **{total_pts}** points from **{len(traj_pos)}** trajectories")
    return traj_pos, traj_time


@app.cell
def _(go, mo, np, traj_pos, traj_time, ui_color):
    # Concatenate all trajectories with NaN separators for single-trace rendering
    _all_x, _all_y, _all_z, _all_c = [], [], [], []
    for _i in range(len(traj_pos)):
        _pts = traj_pos[_i]
        if len(_pts) < 2:
            continue
        _all_x.extend(_pts[:, 0].tolist())
        _all_y.extend(_pts[:, 1].tolist())
        _all_z.extend(_pts[:, 2].tolist())
        _all_x.append(None)
        _all_y.append(None)
        _all_z.append(None)
        if ui_color.value == "Speed":
            _diff = np.diff(_pts, axis=0)
            _speed = np.sqrt(np.sum(_diff**2, axis=1)).tolist()
            _speed.append(_speed[-1])
            _all_c.extend(_speed)
            _all_c.append(None)
        else:
            _all_c.extend(traj_time[_i].tolist())
            _all_c.append(None)

    fig3d = go.Figure(go.Scatter3d(
        x=_all_x, y=_all_y, z=_all_z,
        mode="lines", line=dict(width=1.5, color=_all_c, colorscale="Viridis" if ui_color.value == "Speed" else "Plasma"),
        showlegend=False, hoverinfo="skip",
    ))
    fig3d.update_layout(
        scene=dict(xaxis_title="X [m]", yaxis_title="Y [m]", zaxis_title="Z [m]", aspectmode="data"),
        width=900, height=700, margin=dict(l=0, r=0, t=30, b=0),
    )
    mo.ui.plotly(fig3d)


@app.cell
def _(go, traj_pos):
    # XY projection — single trace
    _all_x, _all_y = [], []
    _sx, _sy, _ex, _ey = [], [], [], []
    for _i in range(len(traj_pos)):
        _pts = traj_pos[_i]
        if len(_pts) < 2:
            continue
        _all_x.extend(_pts[:, 0].tolist())
        _all_y.extend(_pts[:, 1].tolist())
        _all_x.append(None)
        _all_y.append(None)
        _sx.append(_pts[0, 0]); _sy.append(_pts[0, 1])
        _ex.append(_pts[-1, 0]); _ey.append(_pts[-1, 1])

    fig2d = go.Figure()
    fig2d.add_trace(go.Scatter(x=_all_x, y=_all_y, mode="lines",
                               line=dict(width=0.5, color="steelblue"), showlegend=False, hoverinfo="skip"))
    fig2d.add_trace(go.Scatter(x=_sx, y=_sy, mode="markers",
                               marker=dict(size=4, color="green"), showlegend=False, name="start"))
    fig2d.add_trace(go.Scatter(x=_ex, y=_ey, mode="markers",
                               marker=dict(size=4, color="red"), showlegend=False, name="end"))
    fig2d.update_layout(
        xaxis_title="X [m]", yaxis_title="Y [m]", title="XY (green=start, red=end)",
        width=900, height=400, margin=dict(l=50, r=20, t=40, b=50),
    )
    mo.ui.plotly(fig2d)
