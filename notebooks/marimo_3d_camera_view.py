import marimo

__generated_with = "0.23.13"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
    # 3D View of Calibration Target and Cameras

    This notebook visualizes the 3D points of the calibration target
    (`target_on_a_side.txt`) and the estimated camera positions based on the
    `.ori` files.
    The coordinate system is oriented to match OpenPTV: X to the left, Y
    upward, and Z towards the viewer.
    """
    )
    return


@app.cell
def _():
    import os

    import matplotlib.pyplot as plt
    import numpy as np

    return np, os, plt


@app.cell
def _(np, os):
    # Load 3D points
    target_file = "../test_data/test_cavity/cal/target_on_a_side.txt"
    if not os.path.exists(target_file):
        target_file = "test_data/test_cavity/cal/target_on_a_side.txt"

    data = np.loadtxt(target_file)
    point_ids = data[:, 0].astype(int)
    x = data[:, 1]
    y = data[:, 2]
    z = data[:, 3]

    # Load Camera Positions
    cam_centers = []
    cam_names = []
    cam_directions = []

    cal_dir = os.path.dirname(target_file)
    for _i in range(1, 5):
        _ori_file = os.path.join(cal_dir, f"cam{_i}.tif.ori")
        if os.path.exists(_ori_file):
            with open(_ori_file, "r") as _f:
                _lines = _f.readlines()
                _center = list(map(float, _lines[0].strip().split()))
                cam_centers.append(_center)
                cam_names.append(f"Cam {_i}")

                _r1 = list(map(float, _lines[3].strip().split()))
                _r2 = list(map(float, _lines[4].strip().split()))
                _r3 = list(map(float, _lines[5].strip().split()))
                _R = np.array([_r1, _r2, _r3])
                # The optical axis in object space is typically the third row
                # of the rotation matrix. For openptv, R maps object space to
                # image space, so R^T maps image space to object space.
                _optical_axis = _R[2, :]  # direction in object space
                cam_directions.append(_optical_axis)

    cam_centers_arr = np.array(cam_centers)
    cam_directions_arr = np.array(cam_directions)
    return cam_centers_arr, cam_directions_arr, cam_names, point_ids, x, y, z


@app.cell
def _(cam_centers_arr, cam_directions_arr, cam_names, plt, point_ids, x, y, z):
    # Create the 3D plot (matplotlib)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Target points
    ax.scatter(x, y, z, s=16, c="blue", alpha=0.8, label="Target Points")
    for _px, _py, _pz, _pid in zip(x, y, z, point_ids):
        ax.text(_px, _py, _pz, str(_pid), size=8, color="darkblue")

    # Cameras + optical-axis lines
    if len(cam_centers_arr) > 0:
        ax.scatter(
            cam_centers_arr[:, 0],
            cam_centers_arr[:, 1],
            cam_centers_arr[:, 2],
            s=64,
            c="red",
            marker="s",
            label="Cameras",
        )
        for _i in range(len(cam_centers_arr)):
            _start = cam_centers_arr[_i]
            # Draw a line of length 150 towards the target
            _end = _start - 150 * cam_directions_arr[_i]
            ax.plot(
                [_start[0], _end[0]],
                [_start[1], _end[1]],
                [_start[2], _end[2]],
                color="red",
                linewidth=2,
            )
            ax.text(
                _start[0], _start[1], _start[2], cam_names[_i], size=10, color="red"
            )

    # OpenPTV orientation: X to the left (reversed), Y upward, Z towards viewer
    ax.set_xlabel("X (Left)")
    ax.set_ylabel("Y (Up)")
    ax.set_zlabel("Z (Towards Viewer)")
    ax.invert_xaxis()
    ax.set_title("OpenPTV Camera and Target 3D View")
    ax.legend(loc="upper right")
    try:
        ax.set_box_aspect((1, 1, 1))  # roughly equal scaling for x, y, z
    except Exception:
        pass

    ax
    return


if __name__ == "__main__":
    app.run()
