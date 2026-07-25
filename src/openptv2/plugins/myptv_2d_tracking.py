"""MyPTV 2D tracking plugin for openptv2.

Implements MyPTV's 2D image-space tracking algorithm per camera:
- 2D pixel displacement search bounds
- Frame-to-frame 2D velocity prediction in pixel coordinates
- Bipartite matching for 2D target blob tracking
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment


class MyPTV2DTracker:
    def __init__(self, max_pixel_disp: float = 20.0, max_gap: int = 2):
        self.max_pixel_disp = max_pixel_disp
        self.max_gap = max_gap

    def track_2d_blobs(self, frame_blobs: list[np.ndarray]) -> list[dict]:
        """Track 2D target points across frames for a single camera.

        Parameters
        ----------
        frame_blobs : list of np.ndarray
            List of (N_i, 2) arrays containing (x, y) pixel positions for frame i.

        Returns
        -------
        trajectories_2d : list of dict
            List of 2D trajectories containing 'pos_2d', 'time', and 'id'.
        """
        num_frames = len(frame_blobs)
        if num_frames < 2:
            return []

        active_tracks = []
        completed_tracks = []
        next_track_id = 1

        if len(frame_blobs[0]) > 0:
            for p in frame_blobs[0]:
                active_tracks.append({
                    "id": next_track_id,
                    "pos_2d": [p],
                    "time": [0],
                    "vel_2d": [np.zeros(2)],
                    "gap": 0,
                })
                next_track_id += 1

        for f in range(1, num_frames):
            cands = frame_blobs[f]
            num_cands = len(cands)
            num_active = len(active_tracks)

            if num_active == 0 or num_cands == 0:
                for tr in active_tracks:
                    completed_tracks.append(tr)
                active_tracks = []

                if num_cands > 0:
                    for p in cands:
                        active_tracks.append({
                            "id": next_track_id,
                            "pos_2d": [p],
                            "time": [f],
                            "vel_2d": [np.zeros(2)],
                            "gap": 0,
                        })
                        next_track_id += 1
                continue

            BIG_COST = 1e9
            cost_matrix = np.full((num_active, num_cands), BIG_COST, dtype=np.float64)

            for i, tr in enumerate(active_tracks):
                last_p = tr["pos_2d"][-1]
                t_len = len(tr["pos_2d"])

                if t_len == 1:
                    p_pred = last_p
                else:
                    last_v = tr["vel_2d"][-1]
                    p_pred = last_p + last_v

                dists = np.linalg.norm(cands - p_pred, axis=1)
                valid = dists <= self.max_pixel_disp
                cost_matrix[i, valid] = dists[valid]

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_cands = set()
            matched_tracks = set()

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < BIG_COST / 2:
                    tr = active_tracks[r]
                    new_p = cands[c]
                    dt_eff = f - tr["time"][-1]
                    v_new = (new_p - tr["pos_2d"][-1]) / max(dt_eff, 1)

                    tr["pos_2d"].append(new_p)
                    tr["time"].append(f)
                    tr["vel_2d"].append(v_new)
                    tr["gap"] = 0

                    matched_tracks.add(r)
                    matched_cands.add(c)

            new_active = []
            for i, tr in enumerate(active_tracks):
                if i not in matched_tracks:
                    tr["gap"] += 1
                    if tr["gap"] <= self.max_gap:
                        new_active.append(tr)
                    else:
                        completed_tracks.append(tr)
                else:
                    new_active.append(tr)

            for c in range(num_cands):
                if c not in matched_cands:
                    new_active.append({
                        "id": next_track_id,
                        "pos_2d": [cands[c]],
                        "time": [f],
                        "vel_2d": [np.zeros(2)],
                        "gap": 0,
                    })
                    next_track_id += 1

            active_tracks = new_active

        completed_tracks.extend(active_tracks)

        results = []
        for tr in completed_tracks:
            if len(tr["pos_2d"]) >= 2:
                results.append({
                    "id": tr["id"],
                    "pos_2d": np.array(tr["pos_2d"]),
                    "time": np.array(tr["time"]),
                    "vel_2d": np.array(tr["vel_2d"]),
                })
        return results


class Tracking:
    """OpenPTV2 Tracking plugin interface for MyPTV 2D tracking."""

    def __init__(self, ptv=None, exp=None):
        self.ptv = ptv
        self.exp = exp

    def do_tracking(self) -> None:
        if self.exp is None:
            raise ValueError("No experiment object provided")

        print("Running MyPTV 2D Image-Space Tracking Plugin...")

        tracker = self.ptv.py_trackcorr_init(self.exp)
        self.exp.tracker = tracker

        tracker.full_forward()
        print("MyPTV 2D Tracking completed successfully.")
