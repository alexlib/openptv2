import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import zarr

    return mo, np, plt, zarr


@app.cell
def _(mo):
    mo.md("# Trajectory Viewer — TT13 aorta WP1")
    return


@app.cell
def _(mo):
    zarr_path = mo.ui.text(value=r"C:\Users\alex\Downloads\TT13_aorta\wp1\res\run.zarr", label="Zarr store path")
    ui_num = mo.ui.slider(start=10, stop=500, step=10, value=100, label="Trajectories to plot")
    ui_color = mo.ui.dropdown(options=["Speed", "Frame"], value="Speed", label="Color by")
    mo.vstack([zarr_path, mo.hstack([ui_num, ui_color], gap=2)])
    return ui_color, ui_num, zarr_path


@app.cell
def _(mo, np, zarr_path):
    root = zarr.open_group(zarr_path.value, mode="r")
    traj = root["trajectories"]
    pos = np.asarray(traj["pos"])
    time = np.asarray(traj["time"])
    trajid = np.asarray(traj["trajid"])
    unique_ids, counts = np.unique(trajid, return_counts=True)
    mo.md(f"Loaded **{len(unique_ids)}** trajectories, **{len(pos)}** total points")
    return counts, pos, time, trajid, unique_ids


@app.cell
def _(mo, np, plt, pos, time, trajid, ui_color, ui_num, unique_ids, counts):
    top_idx = np.argsort(counts)[::-1][: int(ui_num.value)]
    top_ids = unique_ids[top_idx]

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    sc = None
    for tid in top_ids:
        mask = trajid == tid
        x, y, z = pos[mask]
        if ui_color.value == "Speed":
            diff = np.diff(pos[mask], axis=0)
            speed = np.sqrt(np.sum(diff**2, axis=1))
            speed = np.append(speed, speed[-1])
            sc = ax.scatter(x, y, z, c=speed, cmap="viridis", s=1, alpha=0.6)
        else:
            sc = ax.scatter(x, y, z, c=time[mask], cmap="plasma", s=1, alpha=0.6)

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    if sc is not None:
        plt.colorbar(sc, ax=ax, shrink=0.6, label=ui_color.value)
    plt.gcf()
    return


@app.cell
def _(mo, np, plt, pos, trajid, ui_num, unique_ids, counts):
    top_idx = np.argsort(counts)[::-1][: int(ui_num.value)]
    top_ids = unique_ids[top_idx]

    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    for tid in top_ids:
        mask = trajid == tid
        x, y = pos[mask, 0], pos[mask, 1]
        axes[0].plot(x, y, linewidth=0.5, alpha=0.6)
        axes[0].scatter(x[0], y[0], s=3, c="green", zorder=5)
        axes[0].scatter(x[-1], y[-1], s=3, c="red", zorder=5)
    axes[0].set_xlabel("X [mm]")
    axes[0].set_ylabel("Y [mm]")
    axes[0].set_title("XY (green=start, red=end)")

    for tid in top_ids:
        mask = trajid == tid
        x, z = pos[mask, 0], pos[mask, 2]
        axes[1].plot(x, z, linewidth=0.5, alpha=0.6)
        axes[1].scatter(x[0], z[0], s=3, c="green", zorder=5)
        axes[1].scatter(x[-1], z[-1], s=3, c="red", zorder=5)
    axes[1].set_xlabel("X [mm]")
    axes[1].set_ylabel("Z [mm]")
    axes[1].set_title("XZ (green=start, red=end)")
    plt.tight_layout()
    plt.gcf()
    return


@app.cell
def _(counts, mo, np):
    mo.md(f"""
    **Stats:**
    - Total trajectories: **{len(counts)}**
    - Median length: **{np.median(counts):.0f}** frames
    - Max length: **{np.max(counts)}** frames
    - Mean length: **{np.mean(counts):.1f}** frames
    """)
    return
