"""Frame-by-frame 3D kinematic prediction + Hungarian assignment.

Private helper: the coarse pass of the hybrid_deltat_3d plugin. It is not a
tracking plugin of its own.

- 2-frame velocity-bounded initialization
- Multi-frame polynomial position prediction with acceleration search bounds
- Hungarian / linear assignment distance matching
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from openptv2.plugins._assignment import match_within_radius
from openptv2.tracking_cost import CostWeights, compute_multi_term_cost_matrix


class NearestHungarian3DTracker:
    def __init__(
        self,
        v_max: float = 10.0,
        a_max: float = 50.0,
        max_gap: int = 2,
        dt: float = 0.1,
        cost_weights: CostWeights | None = None,
        max_angle_deg: float | None = None,
    ):
        self.v_max = v_max
        self.a_max = a_max
        self.max_gap = max_gap
        self.dt = dt
        self.cost_weights = cost_weights
        # Cone-of-continuity filter (degrees, applied only to seeded tracks
        # with an established velocity -- a fresh track has no direction to
        # compare against). Unlike trackcorr's cone search, this doesn't
        # gate candidate generation itself; it forbids matches whose implied
        # velocity direction breaks continuity beyond this angle, on top of
        # the existing v_max/a_max distance radius.
        self.max_angle_deg = max_angle_deg

    @staticmethod
    def _new_track(track_id: int, pos: np.ndarray, frame_idx: int) -> dict:
        return {
            "id": track_id,
            "pos": [pos],
            "time": [frame_idx],
            "vel": [np.zeros(3)],
            "gap": 0,
        }

    def _seed_tracks(self, cand_pts, frame_idx, next_track_id):
        """Start one track per candidate point (used on reset and init)."""
        tracks = []
        for p in cand_pts:
            tracks.append(self._new_track(next_track_id, p, frame_idx))
            next_track_id += 1
        return tracks, next_track_id

    def _advance_frame(
        self, f, cand_pts, active_tracks, completed_tracks, next_track_id
    ):
        """Match one frame's candidates against active tracks, return the updated state."""
        num_cands = len(cand_pts)
        num_active = len(active_tracks)

        if num_active == 0 or num_cands == 0:
            completed_tracks.extend(active_tracks)
            new_active, next_track_id = self._seed_tracks(cand_pts, f, next_track_id)
            return new_active, completed_tracks, next_track_id

        # Prediction + search radius for every active track at once. A
        # single-point track has no velocity estimate yet, so it predicts
        # "no motion" and searches the wider v_max ball; a seeded track
        # extrapolates its last velocity and searches a_max.
        last_p = np.array([tr["pos"][-1] for tr in active_tracks])
        last_v = np.array([tr["vel"][-1] for tr in active_tracks])
        seeded = np.fromiter(
            (len(tr["pos"]) > 1 for tr in active_tracks),
            dtype=bool,
            count=num_active,
        )
        pred = np.where(seeded[:, None], last_p + last_v, last_p)
        radius = np.where(seeded, self.a_max, self.v_max)

        if self.cost_weights is not None:
            cost_mat = compute_multi_term_cost_matrix(
                pred_pos=pred,
                cand_pos=cand_pts,
                pred_vel=last_v,
                weights=self.cost_weights,
                dt=self.dt,
            )
        else:
            cost_mat = None

        violates = None
        if self.max_angle_deg is not None and np.any(seeded):
            # disp[r, c] = candidate displacement from track r's last real
            # position (not the extrapolated pred) to candidate c.
            disp = cand_pts[None, :, :] - last_p[:, None, :]
            disp_norm = np.linalg.norm(disp, axis=2)
            v_norm = np.linalg.norm(last_v, axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                cosang = np.sum(disp * last_v[:, None, :], axis=2) / (
                    disp_norm * v_norm[:, None]
                )
            angle_deg = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
            # Only gates seeded tracks with a nonzero last velocity -- a
            # stationary or fresh track has no direction to break.
            gate = seeded[:, None] & (v_norm[:, None] > 1e-9) & (disp_norm > 1e-9)
            violates = gate & (angle_deg > self.max_angle_deg)

        row_ind, col_ind = match_within_radius(
            pred, cand_pts, radius, cost_matrix=cost_mat
        )
        if violates is not None and len(row_ind) > 0:
            # match_within_radius's own in-radius check is purely spatial
            # (raw Euclidean distance), so it doesn't know about the angle
            # cone -- a violating pair could still be the least-bad option
            # available and get returned. Drop those here instead.
            keep = ~violates[row_ind, col_ind]
            row_ind, col_ind = row_ind[keep], col_ind[keep]

        matched_cands = set()
        matched_tracks = set()

        for r, c in zip(row_ind, col_ind):
            tr = active_tracks[r]
            new_p = cand_pts[c]
            dt_eff = (f - tr["time"][-1]) * self.dt
            v_new = (new_p - tr["pos"][-1]) / max(dt_eff, 1e-6)

            tr["pos"].append(new_p)
            tr["time"].append(f)
            tr["vel"].append(v_new)
            tr["gap"] = 0

            matched_tracks.add(r)
            matched_cands.add(c)

        new_active = []
        for i, tr in enumerate(active_tracks):
            if i in matched_tracks:
                new_active.append(tr)
                continue
            tr["gap"] += 1
            if tr["gap"] <= self.max_gap:
                new_active.append(tr)
            else:
                completed_tracks.append(tr)

        for c in range(num_cands):
            if c not in matched_cands:
                new_active.append(self._new_track(next_track_id, cand_pts[c], f))
                next_track_id += 1

        return new_active, completed_tracks, next_track_id

    @staticmethod
    def _finalize(completed_tracks: list[dict]) -> list[dict]:
        results = []
        for tr in completed_tracks:
            if len(tr["pos"]) >= 2:
                results.append(
                    {
                        "id": tr["id"],
                        "pos": np.array(tr["pos"]),
                        "time": np.array(tr["time"]),
                        "vel": np.array(tr["vel"]),
                    }
                )
        return results

    def track_frames(self, frame_particles: list[np.ndarray]) -> list[dict]:
        """Track 3D particles across a list of frame particle arrays.

        Parameters
        ----------
        frame_particles : list of np.ndarray
            List of (N_i, 3) arrays containing 3D positions for frame i.

        Returns
        -------
        trajectories : list of dict
            List of tracked trajectory dictionaries containing 'pos', 'time', and 'id'.
        """
        num_frames = len(frame_particles)
        if num_frames < 2:
            return []

        active_tracks: list[dict[str, Any]] = []
        completed_tracks: list[dict[str, Any]] = []
        next_track_id = 1

        if len(frame_particles[0]) > 0:
            active_tracks, next_track_id = self._seed_tracks(
                frame_particles[0], 0, next_track_id
            )

        for f in range(1, num_frames):
            active_tracks, completed_tracks, next_track_id = self._advance_frame(
                f, frame_particles[f], active_tracks, completed_tracks, next_track_id
            )

        completed_tracks.extend(active_tracks)
        return cast(list[dict[str, Any]], self._finalize(completed_tracks))

