import marimo

__generated_with = "0.23.13"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # 3D View of Calibration Target and Cameras

    This notebook visualizes the 3D points of the calibration target (`target_on_a_side.txt`) and the estimated camera positions based on the `.ori` files.
    The coordinate system is oriented to match OpenPTV: X to the left, Y upward, and Z towards the viewer.
    """)
    return


@app.cell
def _():
    import numpy as np
    import plotly.graph_objects as go
    import os
    import glob

    return go, np, os


@app.cell
def _(np, os):
    # Load 3D points
    target_file = '../test_data/test_cavity/cal/target_on_a_side.txt'
    if not os.path.exists(target_file):
        target_file = 'test_data/test_cavity/cal/target_on_a_side.txt'

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
        _ori_file = os.path.join(cal_dir, f'cam{_i}.tif.ori')
        if os.path.exists(_ori_file):
            with open(_ori_file, 'r') as _f:
                _lines = _f.readlines()
                _center = list(map(float, _lines[0].strip().split()))
                cam_centers.append(_center)
                cam_names.append(f'Cam {_i}')

                _r1 = list(map(float, _lines[3].strip().split()))
                _r2 = list(map(float, _lines[4].strip().split()))
                _r3 = list(map(float, _lines[5].strip().split()))
                _R = np.array([_r1, _r2, _r3])
                # The optical axis in object space is typically the third row of the rotation matrix
                # For openptv, R maps object space to image space, so R^T maps image space to object space
                _optical_axis = _R[2, :] # direction in object space
                cam_directions.append(_optical_axis)

    cam_centers_arr = np.array(cam_centers)
    cam_directions_arr = np.array(cam_directions)
    return cam_centers_arr, cam_directions_arr, cam_names, point_ids, x, y, z


@app.cell
def _(cam_centers_arr, cam_directions_arr, cam_names, go, point_ids, x, y, z):
    # Create the 3D plot
    fig = go.Figure()

    # Add Target Points
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers+text',
        name='Target Points',
        text=point_ids,
        textposition='top center',
        marker=dict(size=4, color='blue', opacity=0.8),
        textfont=dict(size=10, color='darkblue')
    ))

    # Add Cameras
    if len(cam_centers_arr) > 0:
        fig.add_trace(go.Scatter3d(
            x=cam_centers_arr[:, 0],
            y=cam_centers_arr[:, 1],
            z=cam_centers_arr[:, 2],
            mode='markers+text',
            name='Cameras',
            text=cam_names,
            textposition='bottom center',
            marker=dict(size=8, color='red', symbol='square'),
            textfont=dict(size=12, color='red', weight='bold')
        ))

        # Add camera direction lines
        for _i in range(len(cam_centers_arr)):
            _start = cam_centers_arr[_i]
            # Draw a line of length 150 towards the target
            _end = _start - 150 * cam_directions_arr[_i] 
            fig.add_trace(go.Scatter3d(
                x=[_start[0], _end[0]],
                y=[_start[1], _end[1]],
                z=[_start[2], _end[2]],
                mode='lines',
                line=dict(color='red', width=2),
                showlegend=False,
                hoverinfo='none'
            ))

    # Set Layout matching OpenPTV orientation
    # X to the left (reversed), Y upward, Z towards the viewer
    fig.update_layout(
        title='OpenPTV Camera and Target 3D View',
        scene=dict(
            xaxis=dict(title='X (Left)', autorange='reversed', showgrid=True),
            yaxis=dict(title='Y (Up)', showgrid=True),
            zaxis=dict(title='Z (Towards Viewer)', showgrid=True),
            camera=dict(
                up=dict(x=0, y=1, z=0),      # Y is upward
                eye=dict(x=0, y=0, z=2.0)    # Looking from positive Z towards origin
            ),
            aspectmode='data' # Ensures equal scaling for x, y, z so angles aren't distorted
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig
    return


if __name__ == "__main__":
    app.run()
