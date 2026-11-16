"""Run a complex (dense_ghost) case with the unified tracker and save a 3D
interactive trajectory plot (Plotly) to an HTML file.

Usage:
    uv run python notebooks/run_complex_case_3d.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

import openptv2.benchmarking as bm
from openptv2.benchmarking.metrics import _frames_dict

OUT_HTML = Path("/tmp/complex_case_3d.html")


def main():
    spec = bm.ScenarioSpec(
        num_particles=80,
        num_frames=20,
        velocity=1.5,
        velocity_jitter=0.4,
        noise_mm=0.1,
        ghost_ratio=0.35,
        seed=7,
        flow_type="turbulent",
        curvature=0.05,
    )
    true_tracks, frame_gt = bm.generate_scenario(spec)
    rig = bm.make_standard_rig(refract=False)
    work_dir = Path("/tmp/complex_case")
    work_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = bm.write_experiment(rig, frame_gt, work_dir, first_frame=10001)

    t0 = time.perf_counter()
    pred = bm.run_tracker(
        yaml_path, "unified",
        track_overrides=dict(dvxmin=-6, dvxmax=6, dvymin=-6, dvymax=6,
                             dvzmin=-6, dvzmax=6, dacc=6),
    )
    elapsed = time.perf_counter() - t0
    print(f"tracked {len(pred)} trajectories in {elapsed:.2f}s")

    # Build plotly 3D figure: one trace per trajectory, colored by track id.
    fig = go.Figure()
    for pid, pts in pred.items():
        arr = np.asarray(pts, dtype=float)
        if arr.shape[0] < 2:
            continue
        # pts are (frame, x, y, z)
        xs, ys, zs = arr[:, 1], arr[:, 2], arr[:, 3]
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode="lines+markers",
            line=dict(width=2),
            marker=dict(size=2),
            name=f"track {pid}",
            showlegend=False,
        ))
    # ground truth overlay (faint gray)
    for pid, pts in true_tracks.items():
        arr = np.asarray([(p[1], p[2], p[3]) for p in pts], dtype=float)
        # pts in true_tracks are (frame, x,y,z) already? check: true_tracks_abs adds frame+first
        pass
    # ground truth as faint lines
    for pid, pts in list(true_tracks.items())[:0]:
        pass

    # overlay GT as gray scatter
    gt_xs, gt_ys, gt_zs = [], [], []
    for pid, pts in true_tracks.items():
        for p in pts:
            gt_xs.append(p[1]); gt_ys.append(p[2]); gt_zs.append(p[3])
    fig.add_trace(go.Scatter3d(
        x=gt_xs, y=gt_ys, z=gt_zs, mode="markers",
        marker=dict(size=1, color="lightgray", opacity=0.4),
        name="ground truth", showlegend=True,
    ))

    fig.update_layout(
        title=f"Unified tracker (ILP+temporal) — complex dense case: {len(pred)} tracks",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(str(OUT_HTML))
    print(f"saved interactive 3D plot to {OUT_HTML}")


if __name__ == "__main__":
    main()