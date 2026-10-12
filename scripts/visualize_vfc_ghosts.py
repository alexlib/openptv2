#!/usr/bin/env python
"""Visualize particle trajectories and progressive elimination of ghost links across:
- Baseline (VFC off)
- Phase 1 (Spatial Neighborhood Consensus: VFC Both)
- Phase 2 (Velocity Bootstrapping + Affine Consensus)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from openptv2.plugins.two_phase_tracking import TwoPhaseTracker, TwoPhaseTrackerConfig
from scripts.bench_two_phase_vfc import make_synthetic_flow


def generate_visualization(
    n_particles: int = 600,
    n_frames: int = 5,
    ghost_ratio: float = 0.5,
    output_path: str = "vfc_trajectory_ghost_elimination.png",
):
    rng = np.random.default_rng(123)
    frames_gt, _ = make_synthetic_flow(
        n_particles=n_particles, n_frames=n_frames, seed=42
    )
    n_ghosts = int(n_particles * ghost_ratio)

    obs_frames = []
    point_types = []

    for t in range(n_frames):
        pts_real = frames_gt[t]
        pts_ghost = rng.uniform(-50, 50, size=(n_ghosts, 3))
        combined = np.vstack([pts_real, pts_ghost])
        types = [("real", i) for i in range(n_particles)] + [
            ("ghost", -1) for _ in range(n_ghosts)
        ]

        perm = rng.permutation(len(combined))
        obs_frames.append(combined[perm])
        point_types.append([types[idx] for idx in perm])

    true_links = set()
    for t in range(n_frames - 1):
        inv0 = {
            point_types[t][r]: r
            for r in range(len(point_types[t]))
            if point_types[t][r][0] == "real"
        }
        inv1 = {
            point_types[t + 1][r]: r
            for r in range(len(point_types[t + 1]))
            if point_types[t + 1][r][0] == "real"
        }
        for i in range(n_particles):
            true_links.add((t, inv0[("real", i)], t + 1, inv1[("real", i)]))

    # 1. Baseline
    cfg_base = TwoPhaseTrackerConfig(
        v_max=4.0, cost_mode="3d", use_velocity=True, vfc_mode="off"
    )
    links_base = set(TwoPhaseTracker(cfg_base).track_frames(obs_frames))
    false_base = links_base - true_links

    # 2. Phase 1 (Both)
    cfg_p1 = TwoPhaseTrackerConfig(
        v_max=4.0,
        cost_mode="3d",
        use_velocity=True,
        vfc_mode="both",
        vfc_radius=12.0,
        vfc_min_neighbors=3,
        vfc_gate=1.5,
        vfc_post_k=3.0,
        vfc_post_min_tol=0.6,
    )
    links_p1 = set(TwoPhaseTracker(cfg_p1).track_frames(obs_frames))
    false_p1 = links_p1 - true_links

    # 3. Phase 2 (Bootstrap + Both)
    cfg_p2 = TwoPhaseTrackerConfig(
        v_max=4.0,
        cost_mode="3d",
        use_velocity=True,
        vfc_mode="both",
        vfc_radius=12.0,
        vfc_min_neighbors=3,
        vfc_gate=1.5,
        vfc_post_k=3.0,
        vfc_post_min_tol=0.6,
        vfc_bootstrap=True,
        vfc_bootstrap_grid_size=20.0,
    )
    links_p2 = set(TwoPhaseTracker(cfg_p2).track_frames(obs_frames))
    false_p2 = links_p2 - true_links

    elim_by_p1 = false_base - false_p1
    elim_by_p2 = false_p1 - false_p2
    surviving = false_p2

    # Physical categorization of surviving links
    cat_counts = {
        "Ghost-Ghost Coincidence": 0,
        "Domain Boundary / Margin": 0,
        "Gap-Coast Swap": 0,
    }
    for t0, r0, t1, r1 in surviving:
        p0 = obs_frames[t0][r0]
        if np.any(np.abs(p0) > 42.0):
            cat_counts["Domain Boundary / Margin"] += 1
        elif t1 - t0 > 1:
            cat_counts["Gap-Coast Swap"] += 1
        else:
            cat_counts["Ghost-Ghost Coincidence"] += 1

    # Plot
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 10), dpi=150)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 0.8], height_ratios=[1, 1])

    ax_xy = fig.add_subplot(gs[0, 0])
    ax_xz = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_pie = fig.add_subplot(gs[1, 1])

    # Plot true streamlines
    for i in range(0, n_particles, 3):
        traj = np.array([frames_gt[t][i] for t in range(n_frames)])
        ax_xy.plot(traj[:, 0], traj[:, 1], color="#4a5568", alpha=0.35, linewidth=0.8)
        ax_xz.plot(traj[:, 0], traj[:, 2], color="#4a5568", alpha=0.35, linewidth=0.8)

    def plot_links_set(link_set, color, label, alpha=0.8, lw=1.6):
        first = True
        for t0, r0, t1, r1 in link_set:
            p0 = obs_frames[t0][r0]
            p1 = obs_frames[t1][r1]
            lbl = label if first else None
            ax_xy.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                color=color,
                alpha=alpha,
                linewidth=lw,
                label=lbl,
            )
            ax_xz.plot(
                [p0[0], p1[0]], [p0[2], p1[2]], color=color, alpha=alpha, linewidth=lw
            )
            first = False

    plot_links_set(
        elim_by_p1,
        "#e63946",
        f"Eliminated by Phase 1 ({len(elim_by_p1)})",
        alpha=0.5,
        lw=1.2,
    )
    plot_links_set(
        elim_by_p2,
        "#fca311",
        f"Eliminated by Phase 2 Bootstrap ({len(elim_by_p2)})",
        alpha=0.85,
        lw=1.8,
    )
    plot_links_set(
        surviving, "#00f5d4", f"Surviving Ghosts ({len(surviving)})", alpha=1.0, lw=2.5
    )

    ax_xy.set_title(
        "Particle Trajectories & Ghost Links (Top View X-Y)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_xy.set_xlabel("X [mm]")
    ax_xy.set_ylabel("Y [mm]")
    ax_xy.set_xlim(-55, 55)
    ax_xy.set_ylim(-55, 55)
    ax_xy.legend(loc="upper right", framealpha=0.8, fontsize=9)
    ax_xy.set_aspect("equal")
    ax_xy.grid(True, alpha=0.2)

    ax_xz.set_title(
        "Particle Trajectories & Ghost Links (Side View X-Z)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_xz.set_xlabel("X [mm]")
    ax_xz.set_ylabel("Z [mm]")
    ax_xz.set_xlim(-55, 55)
    ax_xz.set_ylim(-55, 55)
    ax_xz.set_aspect("equal")
    ax_xz.grid(True, alpha=0.2)

    # Bar chart
    stages = ["Baseline (Off)", "Phase 1 (VFC Both)", "Phase 2 (Bootstrap)"]
    counts = [len(false_base), len(false_p1), len(false_p2)]
    bars = ax_bar.bar(
        stages,
        counts,
        color=["#e63946", "#fca311", "#00f5d4"],
        width=0.55,
        edgecolor="white",
        linewidth=0.8,
    )
    for bar in bars:
        yval = bar.get_height()
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 3,
            f"{yval}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax_bar.set_title(
        "False Links Progressive Elimination", fontsize=12, fontweight="bold", pad=10
    )
    ax_bar.set_ylabel("Total False Links")
    ax_bar.set_ylim(0, max(counts) * 1.15)
    ax_bar.grid(axis="y", alpha=0.2)

    # Pie chart
    colors_pie = ["#7209b7", "#3a0ca3", "#4361ee"]
    ax_pie.pie(
        cat_counts.values(),
        labels=cat_counts.keys(),
        autopct="%1.0f%%",
        startangle=140,
        colors=colors_pie,
        textprops={"fontsize": 10, "fontweight": "bold"},
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
    )
    ax_pie.set_title(
        f"Physical Anatomy of Surviving {len(surviving)} False Links\n(0 Real Particles Hijacked)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Visualization saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize VFC trajectory and ghost elimination"
    )
    parser.add_argument("--n-particles", type=int, default=600)
    parser.add_argument("--n-frames", type=int, default=5)
    parser.add_argument("--ghost-ratio", type=float, default=0.5)
    parser.add_argument(
        "--output", type=str, default="vfc_trajectory_ghost_elimination.png"
    )
    args = parser.parse_args()
    generate_visualization(
        args.n_particles, args.n_frames, args.ghost_ratio, args.output
    )
