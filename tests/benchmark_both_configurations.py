"""Comprehensive Comparative Benchmark of Configuration A vs Configuration B.

Compares:
- Config A: MyPTV Hybrid Multi-Term + Post-Processing (Cold Start + Gap Relinking)
- Config B: OpenPTV2 Cython Hybrid3D + Post-Processing
- Baseline: Distance-Only Single-Pass Tracker
"""

import os
import shutil
import time
from pathlib import Path

import numpy as np

from openptv2.algorithms.parameters import ControlPar, SequencePar, TrackPar, VolumePar
from openptv2.calibration import Calibration
from openptv2.plugins.myptv_3d_tracking import MyPTV3DTracker
from openptv2.tracker import Tracker
from openptv2.tracking_cost import CostWeights
from openptv2.tracking_metrics import (
    calculate_tracking_metrics,
    generate_synthetic_benchmark_dataset,
)


def read_all_calibration(num_cams, base_path="."):
    cals = []
    for cam in range(num_cams):
        ori_name = f"{base_path}/cal/cam{cam + 1}.tif.ori"
        added_name = f"{base_path}/cal/cam{cam + 1}.tif.addpar"
        cal = Calibration()
        cal.from_file(ori_name, added_name)
        cals.append(cal)
    return cals


def run_configuration_benchmark(title: str, noise: float, gaps: float, spurious: float):
    num_particles = 200
    num_frames = 20

    true_tracks, frame_blobs = generate_synthetic_benchmark_dataset(
        num_particles=num_particles,
        num_frames=num_frames,
        noise_std=noise,
        gap_probability=gaps,
        false_positive_ratio=spurious,
        flow_type="vortex",
        seed=42,
    )

    frame_particle_arrays = [
        np.array(frame_blobs[f], dtype=np.float64) for f in sorted(frame_blobs.keys())
    ]
    total_particles = sum(len(arr) for arr in frame_particle_arrays)

    print("\n" + "=" * 105)
    print(f"--- {title} ---")
    print(
        f"    (Particles: {num_particles}, Frames: {num_frames}, Noise std: {noise}, Gaps: {gaps * 100:.0f}%, Ghost spurious: {spurious * 100:.0f}%)"
    )
    print("=" * 105)
    print(
        f"{'Tracker Configuration':<36} | {'Yield':<7} | {'Precision':<9} | {'Mean Length':<11} | {'RMS Error':<9} | {'FPS':<8} | {'Throughput':<11}"
    )
    print("-" * 105)

    # 1. Baseline Tracker (Distance-Only)
    t0 = time.perf_counter()
    tracker_base = MyPTV3DTracker(v_max=3.0, a_max=1.5, max_gap=0, dt=1.0)
    raw_base = tracker_base.track_frames(frame_particle_arrays)
    t_base = max(time.perf_counter() - t0, 1e-6)

    pred_base = {
        int(tr["id"]): [
            (int(f), float(p[0]), float(p[1]), float(p[2]))
            for f, p in zip(tr["time"], tr["pos"])
        ]
        for tr in raw_base
    }
    m_base = calculate_tracking_metrics(
        true_tracks, pred_base, distance_tolerance=max(0.2, 3.0 * noise)
    )
    m_base.fps = num_frames / t_base
    m_base.particles_per_sec = total_particles / t_base

    print(
        f"{'1. Baseline (Distance-Only)':<36} | {m_base.yield_recall * 100:5.1f}% | {m_base.precision * 100:7.1f}% | {m_base.mean_track_length:9.2f} fr | {m_base.rms_position_error:8.4f} | {m_base.fps:7.1f} | {m_base.particles_per_sec:9.0f} p/s"
    )

    # 2. Configuration A: MyPTV Hybrid Multi-Term (Cost Weights + Gap Buffer max_gap=2)
    weights = CostWeights(w_distance=1.0, w_velocity=0.6, w_acceleration=0.3)
    t0 = time.perf_counter()
    tracker_config_a = MyPTV3DTracker(
        v_max=3.0, a_max=1.5, max_gap=2, dt=1.0, cost_weights=weights
    )
    raw_config_a = tracker_config_a.track_frames(frame_particle_arrays)
    t_config_a = max(time.perf_counter() - t0, 1e-6)

    pred_config_a = {
        int(tr["id"]): [
            (int(f), float(p[0]), float(p[1]), float(p[2]))
            for f, p in zip(tr["time"], tr["pos"])
        ]
        for tr in raw_config_a
    }
    m_config_a = calculate_tracking_metrics(
        true_tracks, pred_config_a, distance_tolerance=max(0.2, 3.0 * noise)
    )
    m_config_a.fps = num_frames / t_config_a
    m_config_a.particles_per_sec = total_particles / t_config_a

    print(
        f"{'2. Config A (Multi-Term + Gap Bridge)':<36} | {m_config_a.yield_recall * 100:5.1f}% | {m_config_a.precision * 100:7.1f}% | {m_config_a.mean_track_length:9.2f} fr | {m_config_a.rms_position_error:8.4f} | {m_config_a.fps:7.1f} | {m_config_a.particles_per_sec:9.0f} p/s"
    )

    # 3. Configuration B: OpenPTV2 Cython Hybrid3D (Compiled C Kernel)
    try:
        from openptv2.algorithms.track_kernels_track3d import track3d_loop_fast

        t0 = time.perf_counter()
        num_parts_arr = [len(pts) for pts in frame_particle_arrays]
        path_x_arr = [
            np.ascontiguousarray(pts, dtype=np.float64)
            if len(pts) > 0
            else np.zeros((0, 3), dtype=np.float64)
            for pts in frame_particle_arrays
        ]
        path_prev_arr = [
            np.full(len(pts), -1, dtype=np.int32) for pts in frame_particle_arrays
        ]
        path_next_arr = [
            np.full(len(pts), -2, dtype=np.int32) for pts in frame_particle_arrays
        ]

        for step in range(1, num_frames - 1):
            f0, f1, f2 = step - 1, step, step + 1
            n0, n1, n2 = num_parts_arr[f0], num_parts_arr[f1], num_parts_arr[f2]
            if n1 > 0 and n2 > 0:
                track3d_loop_fast(
                    n1,
                    path_x_arr[f0],
                    path_prev_arr[f0],
                    n0,
                    path_x_arr[f1],
                    path_prev_arr[f1],
                    path_next_arr[f1],
                    n1,
                    path_x_arr[f2],
                    path_prev_arr[f2],
                    path_next_arr[f2],
                    n2,
                    3.0,
                    3.0,
                    3.0,
                    32,
                    1.5,
                )

        t_config_b = max(time.perf_counter() - t0, 1e-6)

        pred_config_b = {}
        tr_id = 1
        visited = set()
        for f in range(num_frames - 1):
            next_links = path_next_arr[f]
            prev_links = path_prev_arr[f]
            for i in range(num_parts_arr[f]):
                if (f, i) in visited:
                    continue
                if prev_links[i] < 0 and next_links[i] >= 0:
                    curr_f = f
                    curr_i = i
                    pts_tr = []
                    while curr_f < num_frames and curr_i >= 0:
                        visited.add((curr_f, curr_i))
                        p = path_x_arr[curr_f][curr_i]
                        pts_tr.append((curr_f, float(p[0]), float(p[1]), float(p[2])))
                        next_i = (
                            path_next_arr[curr_f][curr_i]
                            if curr_f < len(path_next_arr)
                            else -1
                        )
                        curr_f += 1
                        curr_i = next_i
                    if len(pts_tr) >= 2:
                        pred_config_b[tr_id] = pts_tr
                        tr_id += 1

        m_config_b = calculate_tracking_metrics(
            true_tracks, pred_config_b, distance_tolerance=max(0.2, 3.0 * noise)
        )
        m_config_b.fps = num_frames / t_config_b
        m_config_b.particles_per_sec = total_particles / t_config_b

        print(
            f"{'3. Config B (OpenPTV2 Cython Hybrid3D)':<36} | {m_config_b.yield_recall * 100:5.1f}% | {m_config_b.precision * 100:7.1f}% | {m_config_b.mean_track_length:9.2f} fr | {m_config_b.rms_position_error:8.4f} | {m_config_b.fps:7.1f} | {m_config_b.particles_per_sec:9.0f} p/s"
        )
    except Exception as e:
        print(f"Config B failed: {e}")

    print("=" * 105)


if __name__ == "__main__":
    run_configuration_benchmark(
        title="BENCHMARK 1: MODERATE NOISE & DROPOUT",
        noise=0.05,
        gaps=0.05,
        spurious=0.05,
    )

    run_configuration_benchmark(
        title="BENCHMARK 2: HEAVY NOISE, DROPOUT & GHOST PARTICLES",
        noise=0.20,
        gaps=0.10,
        spurious=0.15,
    )
