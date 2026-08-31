import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from pathlib import Path
    from openptv2.storage import RunStore
    from flowtracks.trajectory import Trajectory
    import flowtracks.smoothing as smoothing

    return Axes3D, Path, RunStore, Trajectory, mo, np, plt, smoothing


@app.cell
def _(mo):
    mo.md("""
    # 🌀 Ilmenau 4-Cam PTV Results & Lagrangian Analysis
    ### Post-Processing with `flowtracks` (PostPTV) & Interactive Matplotlib
    """)
    return


@app.cell
def _(mo):
    default_store = r"C:\Users\alex\Downloads\Illmenau\openptv_illmenau_4cam\res\run.zarr"
    zarr_input = mo.ui.text(value=default_store, label="Zarr Store Path", full_width=True)

    fps_slider = mo.ui.number(start=1.0, stop=1000.0, step=10.0, value=100.0, label="Acquisition FPS [Hz]")
    sg_window = mo.ui.dropdown(options=[5, 7, 9, 11], value=5, label="Savitzky-Golay Window")
    sg_order = mo.ui.dropdown(options=[2, 3], value=2, label="SG Poly Order")

    min_len_slider = mo.ui.slider(start=5, stop=50, step=1, value=5, label="Min Trajectory Length", show_value=True)
    max_trajs_slider = mo.ui.slider(start=10, stop=300, step=10, value=120, label="Max Tracks Displayed", show_value=True)
    
    color_by = mo.ui.dropdown(
        options=["Speed [mm/s]", "Acceleration [mm/s²]", "Time / Frame", "Z Height [m]", "Trajectory ID"],
        value="Speed [mm/s]",
        label="Color Tracks By"
    )
    frame_slider = mo.ui.slider(start=1901, stop=2000, step=1, value=1901, label="Active Frame (3D Cloud Overlay)", show_value=True)
    show_cloud = mo.ui.checkbox(value=True, label="Overlay Frame 3D Particle Cloud")

    controls = mo.vstack([
        zarr_input,
        mo.hstack([fps_slider, sg_window, sg_order], gap=2),
        mo.hstack([min_len_slider, max_trajs_slider, color_by], gap=2),
        mo.hstack([frame_slider, show_cloud], gap=2)
    ])
    controls
    return (
        color_by,
        controls,
        default_store,
        fps_slider,
        frame_slider,
        max_trajs_slider,
        min_len_slider,
        sg_order,
        sg_window,
        show_cloud,
        zarr_input,
    )


@app.cell
def _(
    Path,
    RunStore,
    Trajectory,
    fps_slider,
    min_len_slider,
    mo,
    np,
    sg_order,
    sg_window,
    smoothing,
    zarr_input,
):
    _store_p = Path(zarr_input.value)
    if not _store_p.exists():
        mo.stop(True, mo.md(f"⚠️ **Store path not found:** `{_store_p}`"))

    store = RunStore(_store_p, mode="r")
    _frames_avail = sorted(store.frames())
    _traj_dict = store.trajectories()
    _traj_idx = store.traj_index()

    _tids = _traj_idx.get("trajid", np.array([]))
    _all_tids = _traj_dict.get("trajid", np.array([]))
    _all_pos = _traj_dict.get("pos", np.empty((0, 3)))
    _all_time = _traj_dict.get("time", np.array([]))

    # 1. Construct flowtracks Trajectory objects
    _raw_trajs = []
    for _tid in _tids:
        _idx = np.where(_all_tids == _tid)[0]
        _p = _all_pos[_idx]
        _t = _all_time[_idx]
        if len(_p) >= min_len_slider.value:
            _tr = Trajectory(_p, np.zeros_like(_p), _t, trajid=int(_tid))
            _raw_trajs.append(_tr)

    # 2. Smooth and differentiate with flowtracks Savitzky-Golay filter
    if len(_raw_trajs) > 0:
        _w_size = int(sg_window.value)
        _p_order = int(sg_order.value)
        if _p_order >= _w_size:
            _p_order = _w_size - 1
        smoothed_trajs = smoothing.savitzky_golay(
            _raw_trajs, fps=float(fps_slider.value), window_size=_w_size, order=_p_order
        )
    else:
        smoothed_trajs = []

    _total_track_pts = sum(len(_tr) for _tr in smoothed_trajs)
    
    _stats_callout = mo.hstack([
        mo.stat(label="Frames Range", value=f"{min(_frames_avail)} .. {max(_frames_avail)} ({len(_frames_avail)} frames)"),
        mo.stat(label="Raw Trajectories", value=f"{len(_raw_trajs)}"),
        mo.stat(label="Smoothed Trajectories (flowtracks)", value=f"{len(smoothed_trajs)}"),
        mo.stat(label="Total Linked Points", value=f"{_total_track_pts:,}"),
    ], justify="space-around")
    _stats_callout
    return (
        smoothed_trajs,
        store,
    )


@app.cell
def _(
    color_by,
    frame_slider,
    max_trajs_slider,
    mo,
    np,
    plt,
    show_cloud,
    smoothed_trajs,
    store,
):
    _fig_3d = plt.figure(figsize=(12, 8), dpi=110)
    _ax = _fig_3d.add_subplot(111, projection="3d")

    _curr_frame = int(frame_slider.value)
    
    # 1. Overlay 3D particle correspondences cloud for selected frame
    if show_cloud.value and store.has_correspondences(_curr_frame):
        _c_pos, _ = store.read_correspondences(_curr_frame)
        if len(_c_pos) > 0:
            _ax.scatter(_c_pos[:, 0], _c_pos[:, 1], _c_pos[:, 2],
                       s=8, c="lightgray", alpha=0.35, label=f"Frame {_curr_frame} Cloud ({len(_c_pos)} pts)")

    # 2. Plot flowtracks trajectories
    if len(smoothed_trajs) > 0:
        _sorted_trajs = sorted(smoothed_trajs, key=lambda x: len(x), reverse=True)[:int(max_trajs_slider.value)]
        _cmap = plt.get_cmap("plasma")
        
        # Gather metric for colormap normalization
        if color_by.value == "Speed [mm/s]":
            _all_vals = np.concatenate([np.linalg.norm(_tr.velocity(), axis=1) * 1000.0 for _tr in _sorted_trajs])
        elif color_by.value == "Acceleration [mm/s²]":
            _all_vals = np.concatenate([np.linalg.norm(_tr.accel(), axis=1) * 1000.0 for _tr in _sorted_trajs])
        else:
            _all_vals = np.array([0.0, 1.0])
            
        _vmin, _vmax = (np.percentile(_all_vals, 5), np.percentile(_all_vals, 95)) if len(_all_vals) > 1 else (0.0, 1.0)
        if _vmax <= _vmin: _vmax = _vmin + 1.0

        for _tr in _sorted_trajs:
            _p = _tr.pos()
            if len(_p) < 2: continue
            
            if color_by.value == "Speed [mm/s]":
                _sp = np.linalg.norm(_tr.velocity(), axis=1) * 1000.0
                _colors = _cmap(np.clip((_sp - _vmin) / (_vmax - _vmin), 0, 1))
                for _i in range(len(_p) - 1):
                    _ax.plot(_p[_i:_i+2, 0], _p[_i:_i+2, 1], _p[_i:_i+2, 2], color=_colors[_i], lw=1.6)
            elif color_by.value == "Acceleration [mm/s²]":
                _ac = np.linalg.norm(_tr.accel(), axis=1) * 1000.0
                _colors = _cmap(np.clip((_ac - _vmin) / (_vmax - _vmin), 0, 1))
                for _i in range(len(_p) - 1):
                    _ax.plot(_p[_i:_i+2, 0], _p[_i:_i+2, 1], _p[_i:_i+2, 2], color=_colors[_i], lw=1.6)
            elif color_by.value == "Time / Frame":
                _t = _tr.time()
                _colors = _cmap((_t - 1901) / 99.0)
                for _i in range(len(_p) - 1):
                    _ax.plot(_p[_i:_i+2, 0], _p[_i:_i+2, 1], _p[_i:_i+2, 2], color=_colors[_i], lw=1.6)
            elif color_by.value == "Z Height [m]":
                _colors = _cmap(np.clip((_p[:, 2] + 1.5) / 3.0, 0, 1))
                for _i in range(len(_p) - 1):
                    _ax.plot(_p[_i:_i+2, 0], _p[_i:_i+2, 1], _p[_i:_i+2, 2], color=_colors[_i], lw=1.6)
            else:
                _ax.plot(_p[:, 0], _p[:, 1], _p[:, 2], lw=1.5, alpha=0.85)

            # Mark inception and termination
            _ax.scatter(_p[0, 0], _p[0, 1], _p[0, 2], color="forestgreen", s=14, marker="o")
            _ax.scatter(_p[-1, 0], _p[-1, 1], _p[-1, 2], color="crimson", s=16, marker="^")

    _ax.set_xlabel("X [m]", fontsize=10, labelpad=8)
    _ax.set_ylabel("Y [m]", fontsize=10, labelpad=8)
    _ax.set_zlabel("Z [m]", fontsize=10, labelpad=8)
    _ax.set_title("3D Lagrangian Trajectory Field (flowtracks Savitzky-Golay Smoothed)", fontsize=12, pad=12)
    _ax.grid(True, linestyle="--", alpha=0.5)
    _ax.view_init(elev=22, azim=42)
    plt.tight_layout()
    _fig_3d
    return


@app.cell
def _(frame_slider, max_trajs_slider, mo, plt, show_cloud, smoothed_trajs, store):
    _fig_proj, _axes = plt.subplots(1, 3, figsize=(16, 5), dpi=105)
    _curr_frame_p = int(frame_slider.value)
    
    _c_pos_p = None
    if show_cloud.value and store.has_correspondences(_curr_frame_p):
        _c_pos_p, _ = store.read_correspondences(_curr_frame_p)

    _sorted_p_trajs = sorted(smoothed_trajs, key=lambda x: len(x), reverse=True)[:int(max_trajs_slider.value)]

    # XY Top View
    _ax_xy = _axes[0]
    if _c_pos_p is not None and len(_c_pos_p) > 0:
        _ax_xy.scatter(_c_pos_p[:, 0], _c_pos_p[:, 1], s=4, c="lightgray", alpha=0.3)
    for _tr in _sorted_p_trajs:
        _p = _tr.pos()
        _ax_xy.plot(_p[:, 0], _p[:, 1], lw=1.2, alpha=0.8)
        _ax_xy.scatter(_p[0, 0], _p[0, 1], color="forestgreen", s=8)
        _ax_xy.scatter(_p[-1, 0], _p[-1, 1], color="crimson", s=10)
    _ax_xy.set_xlabel("X [m]")
    _ax_xy.set_ylabel("Y [m]")
    _ax_xy.set_title("XY Projection (Top View)")
    _ax_xy.grid(True, linestyle=":", alpha=0.6)

    # XZ Front View
    _ax_xz = _axes[1]
    if _c_pos_p is not None and len(_c_pos_p) > 0:
        _ax_xz.scatter(_c_pos_p[:, 0], _c_pos_p[:, 2], s=4, c="lightgray", alpha=0.3)
    for _tr in _sorted_p_trajs:
        _p = _tr.pos()
        _ax_xz.plot(_p[:, 0], _p[:, 2], lw=1.2, alpha=0.8)
        _ax_xz.scatter(_p[0, 0], _p[0, 2], color="forestgreen", s=8)
        _ax_xz.scatter(_p[-1, 0], _p[-1, 2], color="crimson", s=10)
    _ax_xz.set_xlabel("X [m]")
    _ax_xz.set_ylabel("Z [m]")
    _ax_xz.set_title("XZ Projection (Front View)")
    _ax_xz.grid(True, linestyle=":", alpha=0.6)

    # YZ Side View
    _ax_yz = _axes[2]
    if _c_pos_p is not None and len(_c_pos_p) > 0:
        _ax_yz.scatter(_c_pos_p[:, 1], _c_pos_p[:, 2], s=4, c="lightgray", alpha=0.3)
    for _tr in _sorted_p_trajs:
        _p = _tr.pos()
        _ax_yz.plot(_p[:, 1], _p[:, 2], lw=1.2, alpha=0.8)
        _ax_yz.scatter(_p[0, 1], _p[0, 2], color="forestgreen", s=8)
        _ax_yz.scatter(_p[-1, 1], _p[-1, 2], color="crimson", s=10)
    _ax_yz.set_xlabel("Y [m]")
    _ax_yz.set_ylabel("Z [m]")
    _ax_yz.set_title("YZ Projection (Side View)")
    _ax_yz.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    _fig_proj
    return


@app.cell
def _(mo, np, plt, smoothed_trajs):
    _fig_stats, (_ax_len, _ax_v, _ax_acc) = plt.subplots(1, 3, figsize=(16, 4.5), dpi=105)

    if len(smoothed_trajs) > 0:
        _lengths = np.array([len(_tr) for _tr in smoothed_trajs])
        _all_vel_mm = np.vstack([_tr.velocity() for _tr in smoothed_trajs]) * 1000.0  # mm/s
        _all_acc_mm = np.vstack([_tr.accel() for _tr in smoothed_trajs]) * 1000.0    # mm/s²
        _all_speed_mm = np.linalg.norm(_all_vel_mm, axis=1)
        _all_acc_mag = np.linalg.norm(_all_acc_mm, axis=1)

        # 1. Trajectory Length Histogram
        _ax_len.hist(_lengths, bins=np.arange(min(_lengths), max(_lengths) + 2), color="royalblue", edgecolor="black", alpha=0.7)
        _ax_len.axvline(np.mean(_lengths), color="red", linestyle="--", label=f"Mean: {np.mean(_lengths):.1f} fr")
        _ax_len.axvline(np.median(_lengths), color="orange", linestyle=":", label=f"Median: {np.median(_lengths):.0f} fr")
        _ax_len.legend()
        _ax_len.set_xlabel("Trajectory Duration [frames]")
        _ax_len.set_ylabel("Count")
        _ax_len.set_title(f"Track Length Distribution (Max: {_lengths.max()} fr)")
        _ax_len.grid(True, linestyle=":", alpha=0.6)

        # 2. Velocity Component PDFs (u, v, w)
        _v_clip = np.percentile(_all_speed_mm, 99)
        _valid_idx = _all_speed_mm < _v_clip
        _ax_v.hist(_all_vel_mm[_valid_idx, 0], bins=35, alpha=0.5, label=f"u (Vx: {np.mean(_all_vel_mm[:, 0]):.1f} mm/s)", color="crimson")
        _ax_v.hist(_all_vel_mm[_valid_idx, 1], bins=35, alpha=0.5, label=f"v (Vy: {np.mean(_all_vel_mm[:, 1]):.1f} mm/s)", color="forestgreen")
        _ax_v.hist(_all_vel_mm[_valid_idx, 2], bins=35, alpha=0.5, label=f"w (Vz: {np.mean(_all_vel_mm[:, 2]):.1f} mm/s)", color="royalblue")
        _ax_v.legend(fontsize=8)
        _ax_v.set_xlabel("Velocity Component [mm/s]")
        _ax_v.set_ylabel("PDF Count")
        _ax_v.set_title(f"Velocity Distribution (Mean Speed: {np.mean(_all_speed_mm):.1f} mm/s)")
        _ax_v.grid(True, linestyle=":", alpha=0.6)

        # 3. Acceleration PDF
        _a_clip = np.percentile(_all_acc_mag, 99)
        _valid_a = _all_acc_mag[_all_acc_mag < _a_clip]
        _ax_acc.hist(_valid_a, bins=35, color="darkcyan", edgecolor="black", alpha=0.7)
        _ax_acc.axvline(np.mean(_valid_a), color="red", linestyle="--", label=f"Mean: {np.mean(_valid_a):.0f} mm/s²")
        _ax_acc.legend()
        _ax_acc.set_xlabel("Acceleration Magnitude [mm/s²]")
        _ax_acc.set_ylabel("Count")
        _ax_acc.set_title("Lagrangian Acceleration Distribution")
        _ax_acc.grid(True, linestyle=":", alpha=0.6)
    else:
        for _a in [_ax_len, _ax_v, _ax_acc]:
            _a.text(0.5, 0.5, "No smoothed trajectories available", ha="center", va="center")

    plt.tight_layout()
    _fig_stats
    return


@app.cell
def _(mo, smoothed_trajs):
    # Single Trajectory Time-Series Inspector
    _traj_options = [int(_tr.trajid()) for _tr in smoothed_trajs] if len(smoothed_trajs) > 0 else [0]
    traj_selector = mo.ui.dropdown(options=_traj_options, value=_traj_options[0], label="Inspect Single Trajectory ID")
    
    mo.vstack([
        mo.md("### 🔍 Single Particle Lagrangian Dynamics Inspector"),
        traj_selector
    ])
    return (traj_selector,)


@app.cell
def _(mo, np, plt, smoothed_trajs, traj_selector):
    _fig_ts, (_ax_pos, _ax_vel, _ax_ac) = plt.subplots(1, 3, figsize=(16, 4.2), dpi=105)
    
    _selected_tid = int(traj_selector.value)
    _target_tr = next((_tr for _tr in smoothed_trajs if int(_tr.trajid()) == _selected_tid), None)
    
    if _target_tr is not None:
        _t = _target_tr.time()
        _p = _target_tr.pos()
        _v = _target_tr.velocity() * 1000.0 # mm/s
        _a = _target_tr.accel() * 1000.0    # mm/s²
        
        # Positions
        _ax_pos.plot(_t, _p[:, 0], label="X(t)", color="crimson", marker=".")
        _ax_pos.plot(_t, _p[:, 1], label="Y(t)", color="forestgreen", marker=".")
        _ax_pos.plot(_t, _p[:, 2], label="Z(t)", color="royalblue", marker=".")
        _ax_pos.set_xlabel("Frame Number")
        _ax_pos.set_ylabel("Position [m]")
        _ax_pos.set_title(f"Trajectory #{_selected_tid} Position vs Time")
        _ax_pos.legend()
        _ax_pos.grid(True, linestyle=":", alpha=0.6)

        # Velocities
        _ax_vel.plot(_t, _v[:, 0], label="u (Vx)", color="crimson", marker=".")
        _ax_vel.plot(_t, _v[:, 1], label="v (Vy)", color="forestgreen", marker=".")
        _ax_vel.plot(_t, _v[:, 2], label="w (Vz)", color="royalblue", marker=".")
        _ax_vel.set_xlabel("Frame Number")
        _ax_vel.set_ylabel("Velocity [mm/s]")
        _ax_vel.set_title("Smoothed Velocities")
        _ax_vel.legend()
        _ax_vel.grid(True, linestyle=":", alpha=0.6)

        # Accelerations
        _ax_ac.plot(_t, _a[:, 0], label="ax", color="crimson", alpha=0.7)
        _ax_ac.plot(_t, _a[:, 1], label="ay", color="forestgreen", alpha=0.7)
        _ax_ac.plot(_t, _a[:, 2], label="az", color="royalblue", alpha=0.7)
        _ax_ac.plot(_t, np.linalg.norm(_a, axis=1), label="|a|", color="black", lw=1.5, linestyle="--")
        _ax_ac.set_xlabel("Frame Number")
        _ax_ac.set_ylabel("Acceleration [mm/s²]")
        _ax_ac.set_title("Smoothed Accelerations")
        _ax_ac.legend()
        _ax_ac.grid(True, linestyle=":", alpha=0.6)
    else:
        for _ax_i in [_ax_pos, _ax_vel, _ax_ac]:
            _ax_i.text(0.5, 0.5, "Select a valid trajectory", ha="center", va="center")

    plt.tight_layout()
    _fig_ts
    return


if __name__ == "__main__":
    app.run()
