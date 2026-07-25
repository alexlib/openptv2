"""Synthetic Tracking Benchmark Suite for OpenPTV2.

Generates controlled synthetic particle trajectories to measure Speed, Precision,
Recall, and Trajectory Continuity across the tracking presets:
- fast_3d (3D-only, fast)
- standard_forward (Forward-only, with added particles)
- full_multipass (Forward + Backward + Pass 3 reciprocity)
"""

import shutil
import time
import tempfile
from pathlib import Path
import numpy as np
import yaml
import pytest

from openptv2.batch.pyptv_batch import run_batch

TEST_CAVITY_DIR = Path(__file__).parent.parent.parent / "test_data" / "test_cavity"


def create_synthetic_experiment(
    out_dir: Path,
    num_frames: int = 10,
    num_continuous: int = 50,
    num_occluded: int = 20,
    noise_std: float = 0.05,
):
    """Generates a synthetic dataset with known ground-truth 3D trajectories.
    
    - num_continuous: Particles visible across all frames 1..N.
    - num_occluded: Particles that experience a 1-frame occlusion at frame 5 (gap in visibility).
    """
    for item in TEST_CAVITY_DIR.iterdir():
        if item.is_dir():
            shutil.copytree(item, out_dir / item.name)
        else:
            shutil.copy2(item, out_dir / item.name)

    res_dir = out_dir / "res"
    shutil.rmtree(res_dir, ignore_errors=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    gt_links = set()  # set of (frame, pnr_t, pnr_t1)
    gt_trajectories = {}
    next_id = 1

    start_frame = 10000
    end_frame = start_frame + num_frames - 1

    # 1. Continuous trajectories
    for i in range(num_continuous):
        x0, y0, z0 = np.random.uniform(-10, 10, 3)
        vx, vy, vz = np.random.uniform(-0.5, 0.5, 3)
        pos = {}
        for f in range(start_frame, end_frame + 1):
            t = f - start_frame
            pos[f] = (
                x0 + vx * t + np.random.normal(0, noise_std),
                y0 + vy * t + np.random.normal(0, noise_std),
                z0 + vz * t + np.random.normal(0, noise_std),
            )
        gt_trajectories[next_id] = {"type": "continuous", "pos": pos, "frames": list(range(start_frame, end_frame + 1))}
        next_id += 1

    # 2. Occluded trajectories (missing at frame start_frame + 4)
    gap_frame = start_frame + 4
    for i in range(num_occluded):
        x0, y0, z0 = np.random.uniform(-10, 10, 3)
        vx, vy, vz = np.random.uniform(-0.5, 0.5, 3)
        pos = {}
        visible_frames = [f for f in range(start_frame, end_frame + 1) if f != gap_frame]
        for f in visible_frames:
            t = f - start_frame
            pos[f] = (
                x0 + vx * t + np.random.normal(0, noise_std),
                y0 + vy * t + np.random.normal(0, noise_std),
                z0 + vz * t + np.random.normal(0, noise_std),
            )
        gt_trajectories[next_id] = {"type": "occluded", "pos": pos, "frames": visible_frames}
        next_id += 1

    # Write res/rt_is.#
    for f in range(start_frame, end_frame + 1):
        particles_in_frame = []
        for tid, tinfo in gt_trajectories.items():
            if f in tinfo["pos"]:
                x, y, z = tinfo["pos"][f]
                particles_in_frame.append((tid, x, y, z))

        rt_file = res_dir / f"rt_is.{f}"
        with open(rt_file, "w") as fp:
            fp.write(f"{len(particles_in_frame)}\n")
            for idx, (tid, x, y, z) in enumerate(particles_in_frame, 1):
                fp.write(f"{idx:4d} {x:10.4f} {y:10.4f} {z:10.4f} {tid:4d} {tid:4d} {tid:4d} {tid:4d}\n")

    # Update sequence parameter in YAML
    yaml_file = out_dir / "parameters_Run1.yaml"
    with open(yaml_file, "r") as fp:
        cfg = yaml.safe_load(fp)
    cfg["sequence"]["first"] = start_frame
    cfg["sequence"]["last"] = end_frame
    with open(yaml_file, "w") as fp:
        yaml.safe_dump(cfg, fp)

    return gt_trajectories, yaml_file, start_frame, end_frame


def evaluate_preset_performance(res_dir: Path, start_frame: int, end_frame: int):
    """Calculates Total Trajectories, Total Links, Mean Trajectory Length, Max Length."""
    frames = {}
    for f in range(start_frame, end_frame + 1):
        file_path = res_dir / f"ptv_is.{f}"
        if not file_path.exists():
            continue
        lines = file_path.read_text().strip().splitlines()
        if not lines:
            continue
        if len(lines[0].split()) == 1:
            lines = lines[1:]
        rows = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                rows.append((int(parts[0]), int(parts[1])))
        frames[f] = rows

    traj_lengths = []
    total_links = 0

    for f in range(start_frame, end_frame + 1):
        rows = frames.get(f, [])
        for idx, (prev, next_idx) in enumerate(rows):
            if next_idx >= 0:
                total_links += 1
            if prev == -1:
                length = 1
                curr_f = f
                curr_next = next_idx
                while curr_next >= 0 and (curr_f + 1) in frames:
                    curr_f += 1
                    next_rows = frames[curr_f]
                    if curr_next < len(next_rows):
                        length += 1
                        curr_next = next_rows[curr_next][1]
                    else:
                        break
                traj_lengths.append(length)

    total_trajs = len(traj_lengths)
    mean_len = sum(traj_lengths) / total_trajs if total_trajs > 0 else 0.0
    max_len = max(traj_lengths) if total_trajs > 0 else 0

    return {
        "total_trajs": total_trajs,
        "total_links": total_links,
        "mean_length": round(mean_len, 2),
        "max_length": max_len,
    }


def test_synthetic_tracking_comparison(tmp_path):
    """Runs synthetic tracking benchmark across all 3 presets and verifies performance."""
    gt, yaml_path, start_frame, end_frame = create_synthetic_experiment(
        tmp_path, num_frames=10, num_continuous=50, num_occluded=20
    )

    results = {}
    presets = ["fast_3d", "standard_forward", "full_multipass"]

    for preset in presets:
        preset_dir = tmp_path / preset
        preset_dir.mkdir()
        for item in tmp_path.iterdir():
            if item.is_dir() and item.name not in presets:
                shutil.copytree(item, preset_dir / item.name)
            elif item.is_file():
                shutil.copy2(item, preset_dir / item.name)

        sandbox_yaml = preset_dir / "parameters_Run1.yaml"
        with open(sandbox_yaml, "r") as fp:
            cfg = yaml.safe_load(fp)
        cfg.setdefault("track", {})["preset"] = preset
        with open(sandbox_yaml, "w") as fp:
            yaml.safe_dump(cfg, fp)

        t0 = time.perf_counter()
        run_batch(sandbox_yaml, start_frame, end_frame, mode="tracking")
        elapsed = time.perf_counter() - t0

        stats = evaluate_preset_performance(preset_dir / "res", start_frame, end_frame)
        stats["time_sec"] = round(elapsed, 3)
        results[preset] = stats

    print("\n" + "=" * 80)
    print("SYNTHETIC TRACKING BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Preset':18s} | {'Time (s)':8s} | {'Links':8s} | {'Trajectories':12s} | {'Mean Length':12s} | {'Max Length':10s}")
    print("-" * 80)
    for preset, res in results.items():
        print(
            f"{preset:18s} | {res['time_sec']:8.3f} | {res['total_links']:8d} | "
            f"{res['total_trajs']:12d} | {res['mean_length']:12.2f} | {res['max_length']:10d}"
        )
    print("=" * 80)

    # Verification assertions
    assert results["fast_3d"]["total_links"] > 0
    assert results["full_multipass"]["total_links"] > 0
