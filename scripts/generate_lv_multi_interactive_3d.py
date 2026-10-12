#!/usr/bin/env python
"""Generate interactive 3D comparison of lv_multi: Fast & Basic (Baseline) vs Extreme Quality (VFC Both + Bootstrap + Affine).
Outputs a rich, standalone interactive 3D WebGL HTML application.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from openptv2.plugins.two_phase_tracking import TwoPhaseTracker, TwoPhaseTrackerConfig
from openptv2.storage import RunStore


def make_line_coords(links, pos_list):
    """Convert a set of links (t0, r0, t1, r1) into NaN-separated WebGL lines."""
    xs, ys, zs = [], [], []
    for t0, r0, t1, r1 in links:
        p0 = pos_list[t0][r0]
        p1 = pos_list[t1][r1]
        xs.extend([p0[0], p1[0], None])
        ys.extend([p0[1], p1[1], None])
        zs.extend([p0[2], p1[2], None])
    return xs, ys, zs


def generate_interactive_html(
    zarr_path: str = "/Users/alex/Documents/Github/openptv-cloud/examples/lv-multi/wp5/res/run.zarr",
    output_html: str = "lv_multi_interactive_3d.html",
):
    print(f"Loading data from: {zarr_path}")
    s = RunStore.open(Path(zarr_path), mode="r")
    frames = range(1, 21)
    pos_list = [
        np.asarray(s.root[f"correspondences/frame_{f:06d}"])[:, :3] for f in frames
    ]
    num_frames = len(frames)
    total_particles = sum(len(p) for p in pos_list)
    print(f"Loaded {num_frames} frames ({total_particles} total 3D detections).")

    # 1. Baseline tracking (Fast & Basic)
    print("Running Baseline Tracker (vfc_mode='off')...")
    cfg_base = TwoPhaseTrackerConfig(
        v_max=1.9,
        cost_mode="3d",
        use_velocity=True,
        vfc_mode="off",
    )
    t0 = time.perf_counter()
    links_base = TwoPhaseTracker(cfg_base).track_frames(pos_list)
    time_base = (time.perf_counter() - t0) * 1000.0 / (num_frames - 1)
    print(f"  Baseline done: {len(links_base)} links ({time_base:.2f} ms/frame)")

    # 2. High Quality tracking (VFC Both + Bootstrap + Affine)
    print("Running High-Quality Tracker (vfc_mode='both', bootstrap, affine)...")
    cfg_hq = TwoPhaseTrackerConfig(
        v_max=1.9,
        cost_mode="3d",
        use_velocity=True,
        vfc_mode="both",
        vfc_radius=12.0,
        vfc_min_neighbors=4,
        vfc_gate=1.0,
        vfc_post_k=3.0,
        vfc_post_min_tol=0.5,
        vfc_bootstrap=True,
        vfc_bootstrap_grid_size=20.0,
        vfc_affine=True,
        vfc_affine_min_neighbors=5,
    )
    t0 = time.perf_counter()
    links_hq = TwoPhaseTracker(cfg_hq).track_frames(pos_list)
    time_hq = (time.perf_counter() - t0) * 1000.0 / (num_frames - 1)
    print(f"  High Quality done: {len(links_hq)} links ({time_hq:.2f} ms/frame)")

    set_base = set(links_base)
    set_hq = set(links_hq)
    common_links = set_base & set_hq
    pruned_links = set_base - set_hq
    added_links = set_hq - set_base

    # Compute speeds
    pruned_speeds = [
        np.linalg.norm(pos_list[t1][r1] - pos_list[t0][r0]) / (t1 - t0)
        for t0, r0, t1, r1 in pruned_links
    ]
    retained_speeds = [
        np.linalg.norm(pos_list[t1][r1] - pos_list[t0][r0]) / (t1 - t0)
        for t0, r0, t1, r1 in common_links
    ]
    med_ret = float(np.median(retained_speeds)) if retained_speeds else 0.0
    med_pruned = float(np.median(pruned_speeds)) if pruned_speeds else 0.0

    print("Building 3D coordinates...")
    com_x, com_y, com_z = make_line_coords(common_links, pos_list)
    prune_x, prune_y, prune_z = make_line_coords(pruned_links, pos_list)
    add_x, add_y, add_z = make_line_coords(added_links, pos_list)

    # Subsample particle dots for background depth
    dots_x, dots_y, dots_z = [], [], []
    for f in range(0, num_frames, 2):
        p = pos_list[f]
        dots_x.extend(p[:, 0])
        dots_y.extend(p[:, 1])
        dots_z.extend(p[:, 2])

    print("Creating Plotly 3D Figure...")
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[
            f"<b>Fast & Basic: Baseline</b> ({time_base:.1f} ms/f | {len(set_base)} links)",
            f"<b>Extreme Quality: Consensus 'Both'</b> ({time_hq:.1f} ms/f | {len(set_hq)} links)",
        ],
        horizontal_spacing=0.03,
    )

    # Particle dots trace
    dots_trace1 = go.Scatter3d(
        x=dots_x[::3],
        y=dots_y[::3],
        z=dots_z[::3],
        mode="markers",
        marker=dict(size=1.2, color="#48cae4", opacity=0.15),
        name="Detected Particles",
        showlegend=True,
    )
    dots_trace2 = go.Scatter3d(
        x=dots_x[::3],
        y=dots_y[::3],
        z=dots_z[::3],
        mode="markers",
        marker=dict(size=1.2, color="#48cae4", opacity=0.15),
        name="Detected Particles",
        showlegend=False,
    )

    # Common tracks
    com_trace1 = go.Scatter3d(
        x=com_x,
        y=com_y,
        z=com_z,
        mode="lines",
        line=dict(color="#0077b6", width=1.5),
        opacity=0.45,
        name=f"Verified Trajectories ({len(common_links):,})",
        showlegend=True,
    )
    com_trace2 = go.Scatter3d(
        x=com_x,
        y=com_y,
        z=com_z,
        mode="lines",
        line=dict(color="#0096c7", width=1.8),
        opacity=0.55,
        name=f"Verified Trajectories ({len(common_links):,})",
        showlegend=False,
    )

    # Pruned spurious jump links (Left Scene)
    prune_trace = go.Scatter3d(
        x=prune_x,
        y=prune_y,
        z=prune_z,
        mode="lines",
        line=dict(color="#ff0055", width=4.0),
        opacity=0.95,
        name=f"Spurious Ghost Jumps Pruned ({len(pruned_links):,})",
        showlegend=True,
    )

    # Newly recovered links (Right Scene)
    add_trace = go.Scatter3d(
        x=add_x,
        y=add_y,
        z=add_z,
        mode="lines",
        line=dict(color="#00f5d4", width=3.5),
        opacity=0.95,
        name=f"Continuity Links Recovered ({len(added_links):,})",
        showlegend=True,
    )

    # Add to Left Subplot (Baseline)
    fig.add_trace(dots_trace1, row=1, col=1)
    fig.add_trace(com_trace1, row=1, col=1)
    fig.add_trace(prune_trace, row=1, col=1)

    # Add to Right Subplot (High Quality)
    fig.add_trace(dots_trace2, row=1, col=2)
    fig.add_trace(com_trace2, row=1, col=2)
    fig.add_trace(add_trace, row=1, col=2)

    scene_dict = dict(
        xaxis=dict(
            title="X [mm]",
            backgroundcolor="#0b0f19",
            gridcolor="#1e293b",
            showbackground=True,
        ),
        yaxis=dict(
            title="Y [mm]",
            backgroundcolor="#0b0f19",
            gridcolor="#1e293b",
            showbackground=True,
        ),
        zaxis=dict(
            title="Z [mm]",
            backgroundcolor="#0b0f19",
            gridcolor="#1e293b",
            showbackground=True,
        ),
        aspectmode="data",
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#070a13",
        plot_bgcolor="#070a13",
        scene=scene_dict,
        scene2=scene_dict,
        margin=dict(l=10, r=10, t=70, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=12, color="#ffffff"),
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="#334155",
            borderwidth=1,
        ),
    )

    # Interactive Controls: Add Dropdown menu for view toggling
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.01,
                y=1.08,
                bgcolor="#1e293b",
                bordercolor="#475569",
                font=dict(color="#ffffff", size=11),
                buttons=[
                    dict(
                        label="Side-by-Side Dual View",
                        method="update",
                        args=[
                            {"visible": [True, True, True, True, True, True]},
                            {
                                "title": f"Real Data lv_multi (wp5): Baseline ({time_base:.1f} ms) vs High Quality Consensus ({time_hq:.1f} ms)"
                            },
                        ],
                    ),
                    dict(
                        label="Highlight Discrepancies Only",
                        method="update",
                        args=[
                            {"visible": [False, False, True, False, False, True]},
                            {
                                "title": f"Discrepancies: {len(pruned_links)} Pruned Ghost Jumps (Red) vs {len(added_links)} Recovered Links (Cyan)"
                            },
                        ],
                    ),
                    dict(
                        label="Clean Trajectories Only",
                        method="update",
                        args=[
                            {"visible": [True, True, False, True, True, False]},
                            {
                                "title": "Clean Coherent Fluid Trajectories (Spurious Links Hidden)"
                            },
                        ],
                    ),
                ],
            )
        ]
    )

    plotly_html = fig.to_html(include_plotlyjs="cdn", full_html=False)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>lv_multi (wp5) Interactive 3D Trajectory Comparison</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: #070a13;
            color: #e2e8f0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            overflow-x: hidden;
        }}
        header {{
            background: linear-gradient(180deg, #0f172a 0%, #070a13 100%);
            border-bottom: 1px solid #1e293b;
            padding: 16px 28px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
        }}
        h1 {{
            font-size: 20px;
            font-weight: 700;
            color: #f8fafc;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .badge {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 4px 8px;
            border-radius: 6px;
            font-weight: 600;
            background: #0284c7;
            color: white;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            padding: 16px 28px;
            background: #0b0f19;
            border-bottom: 1px solid #1e293b;
        }}
        .stat-card {{
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 8px;
            padding: 12px 16px;
        }}
        .stat-label {{
            font-size: 11px;
            color: #94a3b8;
            text-transform: uppercase;
            font-weight: 600;
            margin-bottom: 4px;
        }}
        .stat-value {{
            font-size: 18px;
            font-weight: 700;
            color: #f1f5f9;
        }}
        .stat-sub {{
            font-size: 11px;
            margin-top: 3px;
        }}
        .c-fast {{ color: #38bdf8; }}
        .c-slow {{ color: #f59e0b; }}
        .c-pruned {{ color: #ff0055; }}
        .c-added {{ color: #00f5d4; }}
        #plot-container {{
            width: 100%;
            height: calc(100vh - 160px);
            min-height: 650px;
        }}
        .tip-bar {{
            padding: 8px 28px;
            background: #0f172a;
            font-size: 12px;
            color: #94a3b8;
            display: flex;
            justify-content: space-between;
        }}
    </style>
</head>
<body>
    <header>
        <div>
            <h1>Real Flow Benchmark: <code>lv_multi (wp5)</code> <span class="badge">Two-Phase Tracker</span></h1>
            <p style="font-size: 13px; color: #94a3b8; margin-top: 2px;">
                4-camera 3D PTV dataset (20 frames, ~1,300 particles/frame) &mdash; Interactive 3D trajectory verification
            </p>
        </div>
    </header>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-label">Baseline (Fast & Basic)</div>
            <div class="stat-value c-fast">{time_base:.1f} <span style="font-size: 13px; font-weight: 400;">ms/frame</span></div>
            <div class="stat-sub">{len(set_base):,} links &bull; VFC off</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">High Quality (VFC Both)</div>
            <div class="stat-value c-slow">{time_hq:.1f} <span style="font-size: 13px; font-weight: 400;">ms/frame</span></div>
            <div class="stat-sub">{len(set_hq):,} links &bull; Bootstrap + Affine</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Pruned Ghost Jumps</div>
            <div class="stat-value c-pruned">{len(pruned_links):,} <span style="font-size: 13px; font-weight: 400;">(-{(len(pruned_links) / len(set_base)) * 100:.1f}%)</span></div>
            <div class="stat-sub">Median jump speed: <b>{med_pruned:.2f} mm/f</b> (vs flow {med_ret:.2f} mm/f)</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Unlocked Continuations</div>
            <div class="stat-value c-added">+{len(added_links):,} <span style="font-size: 13px; font-weight: 400;">links</span></div>
            <div class="stat-sub">Recovered by preventing ghost hijack</div>
        </div>
    </div>

    <div class="tip-bar">
        <span><b>Interactions:</b> Click & drag to rotate 3D view &bull; Scroll to zoom &bull; Click legend items to toggle traces</span>
        <span>Use top-left buttons to switch modes</span>
    </div>

    <div id="plot-container">
        {plotly_html}
    </div>
</body>
</html>
"""

    out_p = Path(output_html)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(full_html, encoding="utf-8")
    print(
        f"Generated standalone 3D interactive HTML: {out_p.resolve()} ({out_p.stat().st_size / 1024:.1f} KB)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate interactive 3D comparison for lv_multi"
    )
    parser.add_argument(
        "--zarr",
        type=str,
        default="/Users/alex/Documents/Github/openptv-cloud/examples/lv-multi/wp5/res/run.zarr",
    )
    parser.add_argument("--output", type=str, default="lv_multi_interactive_3d.html")
    args = parser.parse_args()
    generate_interactive_html(args.zarr, args.output)
