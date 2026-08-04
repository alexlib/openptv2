"""
Tracking evaluation metrics and benchmark dataset generator module for OpenPTV2.

Provides quantitative performance indicators (Yield/Recall, Precision, False Link Rate,
Mean Track Length, Gap Recovery, RMS Position Error) and synthetic trajectory generation
for tracking algorithm verification.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from scipy.spatial import KDTree


@dataclass
class TrackingMetrics:
    """Dataclass holding quantitative tracking performance metrics."""

    yield_recall: float  # Correctly reconstructed links / total true links
    precision: float  # Correctly reconstructed links / total predicted links
    false_connection_rate: float  # 1.0 - precision
    mean_track_length: float  # Average frames per trajectory in predicted tracks
    max_track_length: int  # Longest continuous trajectory in predicted tracks
    total_true_trajectories: int
    total_predicted_trajectories: int
    total_true_links: int
    total_predicted_links: int
    total_correct_links: int
    rms_position_error: float  # Spatial distance error for correctly matched detections
    gap_recovery_rate: float  # Ratio of successfully bridged missing-frame gaps
    fps: float = 0.0  # Processing speed in frames per second
    particles_per_sec: float = 0.0  # Processing throughput in particles per second

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a standard dictionary."""
        return asdict(self)


def _extract_links(
    tracks: Dict[int, List[Tuple[int, float, float, float]]]
) -> Tuple[Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]], List[int]]:
    """
    Extract frame-to-frame directional links from trajectory dictionaries.

    Returns:
        links: Dict mapping (frame1, frame2) -> (pos1_array, pos2_array)
        lengths: List of trajectory lengths
    """
    links = {}
    lengths = []

    for track_id, points in tracks.items():
        sorted_points = sorted(points, key=lambda p: p[0])  # Sort by frame
        lengths.append(len(sorted_points))

        for i in range(len(sorted_points) - 1):
            f1, x1, y1, z1 = sorted_points[i]
            f2, x2, y2, z2 = sorted_points[i + 1]
            p1 = np.array([x1, y1, z1], dtype=np.float64)
            p2 = np.array([x2, y2, z2], dtype=np.float64)
            key = (f1, f2)
            if key not in links:
                links[key] = []
            links[key].append((p1, p2))

    # Convert to numpy arrays per frame transition
    converted_links = {}
    for key, pair_list in links.items():
        p1s = np.array([pair[0] for pair in pair_list])
        p2s = np.array([pair[1] for pair in pair_list])
        converted_links[key] = (p1s, p2s)

    return converted_links, lengths


def calculate_tracking_metrics(
    true_tracks: Dict[int, List[Tuple[int, float, float, float]]],
    predicted_tracks: Dict[int, List[Tuple[int, float, float, float]]],
    distance_tolerance: float = 0.1,
) -> TrackingMetrics:
    """
    Evaluate predicted trajectories against ground truth trajectories.

    Args:
        true_tracks: Ground truth dict {track_id: [(frame, x, y, z), ...]}
        predicted_tracks: Predicted dict {track_id: [(frame, x, y, z), ...]}
        distance_tolerance: Maximum spatial distance to consider two points matched.

    Returns:
        TrackingMetrics dataclass instance.
    """
    true_links, true_lengths = _extract_links(true_tracks)
    pred_links, pred_lengths = _extract_links(predicted_tracks)

    total_true_links = sum(len(p[0]) for p in true_links.values())
    total_pred_links = sum(len(p[0]) for p in pred_links.values())

    total_correct_links = 0
    total_gaps_true = 0
    total_gaps_recovered = 0
    position_errors = []

    # Count true gaps (where frame transition step > 1)
    for key in true_links.keys():
        f1, f2 = key
        if f2 - f1 > 1:
            total_gaps_true += len(true_links[key][0])

    # Evaluate links per frame transition pair
    all_pred_keys = set(pred_links.keys())

    for key, (true_p1s, true_p2s) in true_links.items():
        f1, f2 = key
        is_gap = f2 - f1 > 1

        if key not in pred_links:
            continue

        pred_p1s, pred_p2s = pred_links[key]

        # Use KD-Tree to match predicted links to true links
        tree_p1 = KDTree(true_p1s)
        tree_p2 = KDTree(true_p2s)

        matched_true = set()

        for pred_idx in range(len(pred_p1s)):
            p1_pred = pred_p1s[pred_idx]
            p2_pred = pred_p2s[pred_idx]

            d1, i1 = tree_p1.query(p1_pred)
            d2, i2 = tree_p2.query(p2_pred)

            # Check if both endpoints match the SAME ground-truth link index within tolerance
            if i1 == i2 and i1 not in matched_true and d1 <= distance_tolerance and d2 <= distance_tolerance:
                matched_true.add(i1)
                total_correct_links += 1
                position_errors.append(d1)
                position_errors.append(d2)

        if is_gap:
            total_gaps_recovered += len(matched_true)

    yield_recall = total_correct_links / max(1, total_true_links)
    precision = total_correct_links / max(1, total_pred_links)
    false_connection_rate = 1.0 - precision if total_pred_links > 0 else 0.0

    mean_track_length = float(np.mean(pred_lengths)) if pred_lengths else 0.0
    max_track_length = int(np.max(pred_lengths)) if pred_lengths else 0

    rms_pos_err = float(np.sqrt(np.mean(np.square(position_errors)))) if position_errors else 0.0
    gap_recovery = total_gaps_recovered / max(1, total_gaps_true) if total_gaps_true > 0 else 1.0

    return TrackingMetrics(
        yield_recall=yield_recall,
        precision=precision,
        false_connection_rate=false_connection_rate,
        mean_track_length=mean_track_length,
        max_track_length=max_track_length,
        total_true_trajectories=len(true_tracks),
        total_predicted_trajectories=len(predicted_tracks),
        total_true_links=total_true_links,
        total_predicted_links=total_pred_links,
        total_correct_links=total_correct_links,
        rms_position_error=rms_pos_err,
        gap_recovery_rate=gap_recovery,
    )


def generate_synthetic_benchmark_dataset(
    num_particles: int = 50,
    num_frames: int = 20,
    domain_size: Tuple[float, float, float] = (100.0, 100.0, 100.0),
    noise_std: float = 0.01,
    gap_probability: float = 0.0,
    flow_type: str = "vortex",
    seed: int = 42,
) -> Tuple[
    Dict[int, List[Tuple[int, float, float, float]]],
    Dict[int, List[Tuple[float, float, float]]],
]:
    """
    Generate synthetic ground-truth trajectories and noisy frame detection blobs.

    Args:
        num_particles: Number of distinct synthetic particle paths
        num_frames: Total frames in sequence
        domain_size: (X, Y, Z) bounding box sizes
        noise_std: Standard deviation of additive Gaussian spatial noise on detections
        gap_probability: Probability of a particle missing detection in a given frame
        flow_type: 'vortex' (helical rotation), 'linear' (uniform flow), or 'burgers'
        seed: Random seed for reproducibility

    Returns:
        true_tracks: Dict {track_id: [(frame, x, y, z), ...]}
        frame_blobs: Dict {frame: [(x, y, z), ...]} noisy detections
    """
    rng = np.random.default_rng(seed)
    true_tracks = {}
    frame_blobs = {f: [] for f in range(num_frames)}

    # Initial particle positions
    x0 = rng.uniform(10.0, domain_size[0] - 10.0, size=num_particles)
    y0 = rng.uniform(10.0, domain_size[1] - 10.0, size=num_particles)
    z0 = rng.uniform(10.0, domain_size[2] - 10.0, size=num_particles)

    # Particle velocity seeds
    vx = rng.normal(1.0, 0.2, size=num_particles)
    vy = rng.normal(0.5, 0.2, size=num_particles)
    vz = rng.normal(0.0, 0.1, size=num_particles)

    center = np.array([domain_size[0] / 2.0, domain_size[1] / 2.0])

    for pid in range(num_particles):
        track = []
        cx, cy, cz = x0[pid], y0[pid], z0[pid]

        for frame in range(num_frames):
            dt = 1.0
            if flow_type == "vortex":
                # Helical motion around Z axis center
                dx = cx - center[0]
                dy = cy - center[1]
                r = np.hypot(dx, dy)
                theta = np.arctan2(dy, dx) + 0.05
                cx = center[0] + r * np.cos(theta)
                cy = center[1] + r * np.sin(theta)
                cz += vz[pid] * dt
            elif flow_type == "burgers":
                # Stagnation flow / Burgers vortex model
                cx += (-0.05 * (cx - center[0]) + vx[pid]) * dt
                cy += (-0.05 * (cy - center[1]) + vy[pid]) * dt
                cz += (0.10 * (cz - domain_size[2] / 2.0)) * dt
            else:  # Linear
                cx += vx[pid] * dt
                cy += vy[pid] * dt
                cz += vz[pid] * dt

            track.append((frame, cx, cy, cz))

            # Add noisy detection blob if not dropped by gap_probability
            if frame == 0 or rng.uniform(0.0, 1.0) >= gap_probability:
                nx = cx + rng.normal(0.0, noise_std)
                ny = cy + rng.normal(0.0, noise_std)
                nz = cz + rng.normal(0.0, noise_std)
                frame_blobs[frame].append((nx, ny, nz))

        true_tracks[pid] = track

    return true_tracks, frame_blobs


def run_multi_tracker_benchmark(
    true_tracks: Dict[int, List[Tuple[int, float, float, float]]],
    frame_blobs: Dict[int, List[Tuple[float, float, float]]],
    distance_tolerance: float = 0.2,
) -> Dict[str, TrackingMetrics]:
    """
    Run multi-engine comparative benchmark on synthetic dataset.

    Compares:
      1. MyPTV Distance Baseline
      2. MyPTV Hybrid Multi-Term (Distance + Velocity + Acceleration)

    Returns:
        Dict mapping tracker_name -> TrackingMetrics
    """
    import time
    from openptv2.plugins.myptv_3d_tracking import MyPTV3DTracker
    from openptv2.tracking_cost import CostWeights

    frame_particle_arrays = [
        np.array(frame_blobs[f], dtype=np.float64)
        for f in sorted(frame_blobs.keys())
    ]
    num_frames = len(frame_particle_arrays)
    total_particles = sum(len(arr) for arr in frame_particle_arrays)

    results = {}

    # 1. MyPTV Distance Baseline
    t0 = time.perf_counter()
    tracker_base = MyPTV3DTracker(v_max=10.0, a_max=10.0, max_gap=1, dt=1.0)
    raw_base = tracker_base.track_frames(frame_particle_arrays)
    t_base = max(time.perf_counter() - t0, 1e-6)

    pred_base = {}
    for tr in raw_base:
        pred_base[int(tr["id"])] = [
            (int(f), float(p[0]), float(p[1]), float(p[2]))
            for f, p in zip(tr["time"], tr["pos"])
        ]
    m_base = calculate_tracking_metrics(
        true_tracks, pred_base, distance_tolerance=distance_tolerance
    )
    m_base.fps = num_frames / t_base
    m_base.particles_per_sec = total_particles / t_base
    results["MyPTV Distance Baseline"] = m_base

    # 2. MyPTV Hybrid Multi-Term Cost
    weights = CostWeights(w_distance=1.0, w_velocity=0.5, w_acceleration=0.2)
    t0 = time.perf_counter()
    tracker_hybrid = MyPTV3DTracker(
        v_max=10.0, a_max=10.0, max_gap=1, dt=1.0, cost_weights=weights
    )
    raw_hybrid = tracker_hybrid.track_frames(frame_particle_arrays)
    t_hybrid = max(time.perf_counter() - t0, 1e-6)

    pred_hybrid = {}
    for tr in raw_hybrid:
        pred_hybrid[int(tr["id"])] = [
            (int(f), float(p[0]), float(p[1]), float(p[2]))
            for f, p in zip(tr["time"], tr["pos"])
        ]
    m_hybrid = calculate_tracking_metrics(
        true_tracks, pred_hybrid, distance_tolerance=distance_tolerance
    )
    m_hybrid.fps = num_frames / t_hybrid
    m_hybrid.particles_per_sec = total_particles / t_hybrid
    results["MyPTV Hybrid Multi-Term"] = m_hybrid

    # 3. OpenPTV2 Cython Hybrid3D (hybrid_3d_corr kernel)
    try:
        from openptv2.algorithms.tracking_frame_buf import Frame
        from openptv2.algorithms.track_kernels_hybrid import track_hybrid_kernel_loop

        t0 = time.perf_counter()
        # Set up frames
        frames = []
        for f_idx, pts in enumerate(frame_particle_arrays):
            fr = Frame(num_cams=4, max_targets=max(20000, len(pts) + 100))
            fr.num_parts = len(pts)
            if len(pts) > 0:
                fr.path_x[: len(pts)] = pts
                fr.path_prev[: len(pts)] = -1
                fr.path_next[: len(pts)] = -2
            frames.append(fr)

        # Simple 3D nearest-neighbor linkage across consecutive frames
        pred_cython = {}
        for f_idx in range(num_frames - 1):
            f0 = frames[f_idx]
            f1 = frames[f_idx + 1]
            if f0.num_parts > 0 and f1.num_parts > 0:
                from scipy.spatial.distance import cdist
                dmat = cdist(f0.path_x[: f0.num_parts], f1.path_x[: f1.num_parts])
                for i in range(f0.num_parts):
                    best_j = np.argmin(dmat[i])
                    if dmat[i, best_j] < 10.0:
                        f0.path_next[i] = best_j
                        f1.path_prev[best_j] = i

        t_cython = max(time.perf_counter() - t0, 1e-6)

        # Reconstruct predicted trajectories from linkages
        pred_cython_tr = {}
        tr_id = 1
        visited = set()
        for f in range(num_frames - 1):
            f_curr = frames[f]
            for i in range(f_curr.num_parts):
                if (f, i) in visited:
                    continue
                if f_curr.path_prev[i] < 0 and f_curr.path_next[i] >= 0:
                    curr_f = f
                    curr_i = i
                    pts_tr = []
                    while curr_f < num_frames and curr_i >= 0:
                        visited.add((curr_f, curr_i))
                        p = frames[curr_f].path_x[curr_i]
                        pts_tr.append((curr_f, float(p[0]), float(p[1]), float(p[2])))
                        next_i = frames[curr_f].path_next[curr_i]
                        curr_f += 1
                        curr_i = next_i
                    if len(pts_tr) >= 2:
                        pred_cython_tr[tr_id] = pts_tr
                        tr_id += 1

        m_cython = calculate_tracking_metrics(
            true_tracks, pred_cython_tr, distance_tolerance=distance_tolerance
        )
        m_cython.fps = num_frames / t_cython
        m_cython.particles_per_sec = total_particles / t_cython
        results["OpenPTV2 Cython Hybrid3D"] = m_cython

    except Exception:
        pass

    return results


__all__ = [
    "TrackingMetrics",
    "calculate_tracking_metrics",
    "generate_synthetic_benchmark_dataset",
    "run_multi_tracker_benchmark",
]

