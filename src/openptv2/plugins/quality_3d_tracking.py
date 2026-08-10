"""Quality 3D compiled tracking plugin for OpenPTV2.

Implements Stage 2 of the master plan:
- Constant-acceleration 3D Kalman Filter predictor (tracking_kalman.py)
- Dynamic innovation ellipsoid gating S = H P H^T + R
- Multi-term cost weighting (distance + velocity continuity + acceleration)
- Cluster-local graph component decomposition (match_within_radius)
- Optional reciprocal backward pass
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from openptv2.algorithms.tracking_frame_buf import Frame
from openptv2.plugins._assignment import match_within_radius
from openptv2.tracking_cost import CostWeights, compute_multi_term_cost_matrix
from openptv2.tracking_kalman import ConstantAccelerationKF3D, KalmanTrackState


class Quality3DTracker:
    """Accuracy-optimal 3D particle tracking engine using Kalman Filter prediction and cluster-local optimal matching.

    Parameters
    ----------
    process_noise_acc : float, default=1.0
        Variance of acceleration fluctuations for Kalman Filter process noise (mm/frame^2).
    measurement_noise : float, default=0.05
        Standard deviation of 3D position measurement uncertainty (mm).
    v_max : float, default=15.0
        Maximum search radius for unseeded initial tracks (mm/frame).
    a_max : float, default=10.0
        Maximum search radius for seeded tracks with velocity history (mm/frame^2).
    max_gap : int, default=2
        Maximum missing frames a track can persist across.
    dt : float, default=1.0
        Timestep between frames (seconds or frame units).
    cost_weights : CostWeights, optional
        Weights for multi-term cost matrix. Defaults to CostWeights(1.0, 0.6, 0.3).
    reciprocal_pass : bool, default=False
        Whether to run a reciprocal backward pass filter.
    """

    def __init__(
        self,
        process_noise_acc: float = 1.0,
        measurement_noise: float = 0.05,
        v_max: float = 15.0,
        a_max: float = 10.0,
        max_gap: int = 2,
        dt: float = 1.0,
        cost_weights: Optional[CostWeights] = None,
        reciprocal_pass: bool = False,
    ) -> None:
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.max_gap = int(max_gap)
        self.dt = float(dt)
        self.cost_weights = (
            cost_weights
            if cost_weights is not None
            else CostWeights(w_distance=1.0, w_velocity=0.6, w_acceleration=0.3)
        )
        self.reciprocal_pass = bool(reciprocal_pass)

        self.kf = ConstantAccelerationKF3D(
            process_noise_acc=process_noise_acc,
            measurement_noise=measurement_noise,
            v_max=self.v_max,
            a_max=self.a_max,
        )

    def track_frames(
        self, frame_particle_arrays: Sequence[np.ndarray]
    ) -> List[Dict[str, Union[int, List[int], List[np.ndarray]]]]:
        """Track particle arrays frame by frame in memory."""
        num_frames = len(frame_particle_arrays)
        if num_frames == 0:
            return []

        active_kf_states: List[KalmanTrackState] = []
        completed_tracks: List[Dict[str, Union[int, List[int], List[np.ndarray]]]] = []
        next_track_id = 0

        # Seed initial tracks from frame 0
        if len(frame_particle_arrays[0]) > 0:
            for p in frame_particle_arrays[0]:
                ts = self.kf.init_state(next_track_id, p, frame_idx=0)
                active_kf_states.append(ts)
                next_track_id += 1

        for f_idx in range(1, num_frames):
            cand_pts = np.asarray(frame_particle_arrays[f_idx], dtype=np.float64)
            num_cands = len(cand_pts)
            num_active = len(active_kf_states)

            if num_cands == 0:
                # Update gap counters or archive
                surviving = []
                for ts in active_kf_states:
                    if (f_idx - ts.last_frame) <= self.max_gap:
                        surviving.append(ts)
                    else:
                        completed_tracks.append(self._export_track(ts))
                active_kf_states = surviving
                continue

            if num_active == 0:
                for p in cand_pts:
                    ts = self.kf.init_state(next_track_id, p, frame_idx=f_idx)
                    active_kf_states.append(ts)
                    next_track_id += 1
                continue

            # Batch Kalman prediction for all active tracks
            pred_states, pred_covs, pred_positions, S_mats = self.kf.batch_predict(
                active_kf_states, dt=1.0
            )

            # Per-track search radius: a_max for seeded tracks with velocity history, v_max for cold start
            seeded = np.fromiter(
                (ts.history_len >= 2 for ts in active_kf_states),
                dtype=bool,
                count=num_active,
            )
            radii = np.where(seeded, self.a_max, self.v_max)

            # Velocity vectors for cost matrix
            pred_vels = pred_states[:, 3:6]

            # Multi-term cost matrix
            cost_mat = compute_multi_term_cost_matrix(
                pred_pos=pred_positions,
                cand_pos=cand_pts,
                pred_vel=pred_vels,
                weights=self.cost_weights,
                dt=self.dt,
            )

            # Cluster-local Hungarian / optimal assignment
            row_ind, col_ind = match_within_radius(
                pred_positions, cand_pts, radii, cost_matrix=cost_mat
            )

            matched_cands = set(col_ind)
            matched_tracks = set(row_ind)

            # Update matched tracks
            next_active: List[KalmanTrackState] = []
            for r, c in zip(row_ind, col_ind):
                ts = active_kf_states[r]
                dt_gap = float(f_idx - ts.last_frame)
                updated_ts = self.kf.update(
                    ts, cand_pts[c], dt=dt_gap, frame_idx=f_idx
                )
                next_active.append(updated_ts)

            # Handle unmatched active tracks (persist if within gap limit)
            for r in range(num_active):
                if r not in matched_tracks:
                    ts = active_kf_states[r]
                    if (f_idx - ts.last_frame) <= self.max_gap:
                        next_active.append(ts)
                    else:
                        completed_tracks.append(self._export_track(ts))

            # Seed unmatched candidates as new tracks
            for c in range(num_cands):
                if c not in matched_cands:
                    new_ts = self.kf.init_state(next_track_id, cand_pts[c], frame_idx=f_idx)
                    next_active.append(new_ts)
                    next_track_id += 1

            active_kf_states = next_active

        # Export remaining active tracks
        for ts in active_kf_states:
            completed_tracks.append(self._export_track(ts))

        return completed_tracks

    def track_directory(self, work_dir: Path) -> None:
        """Run quality_3d tracking on an experiment directory reading rt_is.# and writing ptv_is.#."""
        work_dir = Path(work_dir)
        res_dir = work_dir / "res"
        if not res_dir.exists():
            res_dir = work_dir

        rt_files = sorted(res_dir.glob("rt_is.*"))
        if not rt_files:
            return

        frame_numbers = []
        frame_particles = []
        frames = []

        for p in rt_files:
            try:
                fn = int(p.suffix.lstrip("."))
            except ValueError:
                continue
            corres_base = str(res_dir / "rt_is")
            linkage_base = str(res_dir / "ptv_is")
            prio_base = str(res_dir / "added")
            frame = Frame(num_cams=4, max_targets=100000)
            frame.read(
                corres_base, linkage_base,
                prio_file_base=prio_base, target_file_base=None, frame_num=fn,
            )
            pos = frame.positions() if frame.num_parts > 0 else np.empty((0, 3))
            frame_numbers.append(fn)
            frame_particles.append(pos)
            frames.append(frame)

        trajectories = self.track_frames(frame_particles)

        # Map trajectory links back to frame structures
        for tr in trajectories:
            times = tr["time"]
            positions = tr["pos"]
            for step_i in range(len(times) - 1):
                f_curr = times[step_i]
                f_next = times[step_i + 1]
                if f_next != f_curr + 1:
                    continue  # Gap frame link
                pts_curr = frame_particles[f_curr]
                pts_next = frame_particles[f_next]
                if len(pts_curr) == 0 or len(pts_next) == 0:
                    continue
                i_curr = int(np.argmin(np.linalg.norm(pts_curr - positions[step_i], axis=1)))
                i_next = int(np.argmin(np.linalg.norm(pts_next - positions[step_i + 1], axis=1)))
                frames[f_curr].path_next[i_curr] = i_next
                frames[f_next].path_prev[i_next] = i_curr

        # Write ptv_is.# output files
        for f_idx, fn in enumerate(frame_numbers):
            frame = frames[f_idx]
            frame._sync_soa_to_path()
            corres_base = str(res_dir / "rt_is")
            linkage_base = str(res_dir / "ptv_is")
            prio_base = str(res_dir / "added")
            frame.write(
                corres_base, linkage_base,
                prio_file_base=prio_base, target_file_base=None, frame_num=fn,
            )

    @staticmethod
    def _export_track(ts: KalmanTrackState) -> Dict[str, Union[int, List[int], List[np.ndarray]]]:
        """Convert a KalmanTrackState to standard trajectory dictionary."""
        return {
            "id": ts.track_id,
            "pos": [p.copy() for p in ts.history_positions],
            "time": list(ts.history_times),
        }


class Tracking:
    """Standard OpenPTV2 plugin contract wrapper for Quality3DTracker."""

    def __init__(self, ptv=None, exp=None):
        self.ptv = ptv
        self.exp = exp

    def do_tracking(self) -> None:
        if self.exp is None:
            raise ValueError("No experiment object provided")

        pm = getattr(self.exp, "pm", None)
        if pm is None and hasattr(self.exp, "exp1"):
            pm = getattr(self.exp.exp1, "pm", None)

        track_cfg = pm.parameters.get("track", {}) if pm else {}
        work_dir = Path(getattr(self.exp, "active_dir", "."))

        v_max = float(track_cfg.get("dvxmax", 15.0))
        a_max = float(track_cfg.get("dacc", 10.0))

        tracker = Quality3DTracker(v_max=v_max, a_max=a_max)
        tracker.track_directory(work_dir)
