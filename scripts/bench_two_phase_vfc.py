"""Benchmark script for Spatial Neighborhood Consensus (VFC) in TwoPhaseTracker.

Compares:
  - Baseline: vfc_mode="off"
  - Option A (Gate): vfc_mode="pre", vfc_gate=...
  - Option A (Soft Weight): vfc_mode="pre", vfc_weight=...
  - Option B (Post-Assignment Fast-VFC): vfc_mode="post"
  - Both: vfc_mode="both"

Evaluates Speed (ms/frame) and Performance (Precision, Recall, F1) across
different ghost particle densities.

Usage:
    uv run python scripts/bench_two_phase_vfc.py [--n-particles 1000] [--n-frames 6]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from openptv2.plugins.two_phase_tracking import TwoPhaseTracker, TwoPhaseTrackerConfig


def make_synthetic_flow(
    n_particles: int = 1000,
    n_frames: int = 6,
    dt: float = 1.0,
    domain_size: float = 100.0,
    v_base: float = 2.0,
    noise_sigma: float = 0.05,
    seed: int = 42,
) -> tuple[list[np.ndarray], set[tuple[int, int, int, int]]]:
    """Generate ground truth synthetic trajectories following a smooth vortex flow.

    Returns:
        frames_gt: list of (N, 3) arrays of true particle positions.
        true_links: set of (t0, row0, t1, row1) tuples representing ground truth links.
    """
    rng = np.random.default_rng(seed)
    # Initial particle positions uniform in [-domain_size/2, domain_size/2]^3
    pos = rng.uniform(-domain_size / 2, domain_size / 2, size=(n_particles, 3))

    frames_gt = [pos.copy()]
    true_links = set()

    for t in range(n_frames - 1):
        x = pos[:, 0]
        y = pos[:, 1]
        r = np.sqrt(x**2 + y**2) + 1.0
        # Smooth circular vortex velocity: v_theta = v_base * r / (1 + (r/30)^2)
        v_mag = v_base * (r / 30.0) / (1.0 + (r / 30.0) ** 2)
        vx = -v_mag * (y / r)
        vy = v_mag * (x / r)
        vz = 0.2 * np.sin(x / 15.0)

        vel = np.column_stack([vx, vy, vz])
        noise = rng.normal(0, noise_sigma, size=pos.shape)
        new_pos = pos + vel * dt + noise
        frames_gt.append(new_pos.copy())

        for i in range(n_particles):
            true_links.add((t, i, t + 1, i))

        pos = new_pos

    return frames_gt, true_links


def inject_ghost_particles(
    frames_gt: list[np.ndarray],
    ghost_ratio: float = 0.5,
    domain_size: float = 100.0,
    seed: int = 123,
) -> tuple[list[np.ndarray], set[tuple[int, int, int, int]]]:
    """Inject randomly placed ghost particles into frames and return observed frames + true links.

    Parameters:
        frames_gt: list of (N, 3) true particle positions.
        ghost_ratio: N_ghosts / N_real. (0.5 means 50% extra particles are ghosts).
    """
    if ghost_ratio <= 0.0:
        n_p = len(frames_gt[0])
        true_links = {
            (t, i, t + 1, i) for t in range(len(frames_gt) - 1) for i in range(n_p)
        }
        return [f.copy() for f in frames_gt], true_links

    rng = np.random.default_rng(seed)
    n_real = len(frames_gt[0])
    n_ghost = int(n_real * ghost_ratio)

    obs_frames = []
    # Real particles will be at the beginning of each frame array (0..n_real-1)
    # Permuting indices to test general ordering:
    gt_to_obs = []
    for f_idx, gt in enumerate(frames_gt):
        ghosts = rng.uniform(-domain_size / 2, domain_size / 2, size=(n_ghost, 3))
        combined = np.vstack([gt, ghosts])
        perm = rng.permutation(len(combined))
        obs_frames.append(combined[perm])

        # Invert permutation to locate where true particle i ended up
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(len(perm))
        gt_to_obs.append(inv_perm[:n_real])

    true_links = set()
    for t in range(len(frames_gt) - 1):
        for i in range(n_real):
            r0 = int(gt_to_obs[t][i])
            r1 = int(gt_to_obs[t + 1][i])
            true_links.add((t, r0, t + 1, r1))

    return obs_frames, true_links


def evaluate_tracking(
    proposed_links: list[tuple[int, int, int, int]],
    true_links: set[tuple[int, int, int, int]],
) -> dict[str, float]:
    """Calculate Precision, Recall, and F1 score."""
    prop_set = set(proposed_links)
    tp = len(prop_set & true_links)
    fp = len(prop_set - true_links)
    fn = len(true_links - prop_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision * 100.0,
        "recall": recall * 100.0,
        "f1": f1 * 100.0,
    }


def run_benchmark(n_particles: int = 1000, n_frames: int = 5, reps: int = 3):
    print("=" * 80)
    print("BENCHMARK: Two-Phase Tracker with Spatial Consensus (VFC)")
    print(f"Particles: {n_particles} | Frames: {n_frames} | Repetitions: {reps}")
    print("=" * 80)

    frames_gt, _ = make_synthetic_flow(n_particles=n_particles, n_frames=n_frames)

    ghost_levels = [0.0, 0.5, 1.0, 2.0]  # 0%, 50%, 100%, 200% ghost particles

    configs = {
        "Baseline (VFC off)": TwoPhaseTrackerConfig(
            v_max=4.0,
            cost_mode="3d",
            use_velocity=True,
            vfc_mode="off",
        ),
        "Option A (Pre-Gate 1.5mm)": TwoPhaseTrackerConfig(
            v_max=4.0,
            cost_mode="3d",
            use_velocity=True,
            vfc_mode="pre",
            vfc_radius=12.0,
            vfc_min_neighbors=3,
            vfc_gate=1.5,
        ),
        "Option A (Soft Weight 2.0)": TwoPhaseTrackerConfig(
            v_max=4.0,
            cost_mode="3d",
            use_velocity=True,
            vfc_mode="pre",
            vfc_radius=12.0,
            vfc_min_neighbors=3,
            vfc_weight=2.0,
        ),
        "Option B (Post-Fast-VFC)": TwoPhaseTrackerConfig(
            v_max=4.0,
            cost_mode="3d",
            use_velocity=True,
            vfc_mode="post",
            vfc_radius=12.0,
            vfc_min_neighbors=3,
            vfc_post_k=3.0,
            vfc_post_min_tol=0.6,
        ),
        "Both (Pre-Gate + Post-VFC)": TwoPhaseTrackerConfig(
            v_max=4.0,
            cost_mode="3d",
            use_velocity=True,
            vfc_mode="both",
            vfc_radius=12.0,
            vfc_min_neighbors=3,
            vfc_gate=1.5,
            vfc_post_k=3.0,
            vfc_post_min_tol=0.6,
        ),
        "Phase 2 (Bootstrap + Both)": TwoPhaseTrackerConfig(
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
        ),
        "Phase 2 (Bootstrap + Affine)": TwoPhaseTrackerConfig(
            v_max=4.0,
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
        ),
    }

    for ghost_ratio in ghost_levels:
        ghost_pct = int(ghost_ratio * 100)
        n_total = n_particles + int(n_particles * ghost_ratio)
        print(
            f"\n--- Seeding Density: {n_particles} real + {int(n_particles * ghost_ratio)} ghosts ({ghost_pct}% ghost ratio, {n_total} pts/frame) ---"
        )
        print(
            f"{'Method':<30} | {'Time (ms/f)':<11} | {'Precision':<10} | {'Recall':<8} | {'F1 Score':<8} | {'False Links':<11}"
        )
        print("-" * 88)

        obs_frames, true_links = inject_ghost_particles(
            frames_gt, ghost_ratio=ghost_ratio
        )

        for name, cfg in configs.items():
            tracker = TwoPhaseTracker(cfg)

            times = []
            links = []
            for _ in range(reps):
                t0 = time.perf_counter()
                links = tracker.track_frames(obs_frames)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0 / (n_frames - 1))

            avg_time = float(np.mean(times))
            metrics = evaluate_tracking(links, true_links)

            print(
                f"{name:<30} | {avg_time:9.2f} ms | "
                f"{metrics['precision']:9.1f}% | "
                f"{metrics['recall']:7.1f}% | "
                f"{metrics['f1']:7.1f}% | "
                f"{metrics['fp']:11d}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Two-Phase Spatial Consensus (VFC)"
    )
    parser.add_argument(
        "--n-particles", type=int, default=800, help="Number of true particles"
    )
    parser.add_argument("--n-frames", type=int, default=5, help="Number of frames")
    parser.add_argument(
        "--reps", type=int, default=3, help="Benchmark repetitions per case"
    )
    args = parser.parse_args()

    run_benchmark(n_particles=args.n_particles, n_frames=args.n_frames, reps=args.reps)
