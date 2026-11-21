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
        num_particles=1000,
        num_frames=40,
        velocity=1.2,
        velocity_jitter=0.25,
        noise_mm=0.05,
        ghost_ratio=0.25,
        seed=7,
        flow_type="turbulent",
        curvature=0.03,
    )
    true_tracks, frame_gt = bm.generate_scenario(spec)
    rig = bm.make_standard_rig(refract=False)
    work_dir = Path("test_data/complex_dense_turbulent")
    work_dir.mkdir(parents=True, exist_ok=True)
    work_dir = work_dir.resolve()
    yaml_path = bm.write_experiment(rig, frame_gt, work_dir, first_frame=10001)
    # Save scenario spec for reproducibility / future benchmarking.
    import json
    (work_dir / "scenario.json").write_text(json.dumps(spec.__dict__, indent=2, default=str))

    t0 = time.perf_counter()
    pred = bm.run_tracker(
        yaml_path, "unified",
        track_overrides=dict(dvxmin=-6, dvxmax=6, dvymin=-6, dvymax=6,
                             dvzmin=-6, dvzmax=6, dacc=6),
    )
    elapsed = time.perf_counter() - t0
    first_frame = 10001
    true_tracks_abs = {
        pid: [(f + first_frame, x, y, z) for f, x, y, z in pts]
        for pid, pts in true_tracks.items()
    }
    eps = max(0.5, 2.0 * spec.noise_mm)
    jump_threshold_mm = 2.5
    from openptv2.benchmarking.metrics import _frames_dict, _match_frame
    import numpy as np
    true_by_frame = _frames_dict(true_tracks_abs)
    pred_by_frame = _frames_dict(pred)
    all_frames = sorted(set(true_by_frame) | set(pred_by_frame))
    matched_true = 0
    total_true = 0
    total_pred = 0
    for f in all_frames:
        tm = true_by_frame.get(f, {})
        pm = pred_by_frame.get(f, {})
        m = _match_frame(tm, pm, eps)
        matched_true += len(m)
        total_true += len(tm)
        total_pred += len(pm)
    link_yield = matched_true / total_true if total_true else 0.0
    precision = matched_true / total_pred if total_pred else 0.0
    lengths = [len(pts) for pts in pred.values()]
    median_len = float(np.median(lengths)) if lengths else 0.0
    print(f"n_tracks={len(pred)} len_min={min(lengths)} len_max={max(lengths)} len_mean={np.mean(lengths):.1f}")
    # jump rate
    n_jumps = 0; n_trans = 0
    for pts in pred.values():
        arr = np.asarray(pts, float)
        if len(arr) < 3:
            continue
        for i in range(1, len(arr) - 1):
            v1 = arr[i] - arr[i-1]; v2 = arr[i+1] - arr[i]
            if np.linalg.norm(v2 - v1) > jump_threshold_mm:
                n_jumps += 1
            n_trans += 1
    jump_rate = n_jumps / n_trans if n_trans else 0.0
    id_metrics = bm.compute_identity_metrics(true_tracks_abs, pred, eps=eps)
    id_metrics = bm.compute_identity_metrics(true_tracks_abs, pred, eps=eps)
    print(f"tracked {len(pred)} trajectories in {elapsed:.2f}s")
    print(f"GT particles: {len(true_tracks)}")
    print(f"link_yield={link_yield:.3f} precision={precision:.3f} median_len={median_len:.1f} jump_rate={jump_rate:.3f}")
    print(f"completeness={id_metrics.completeness:.3f} purity={id_metrics.purity:.3f}")

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