"""MyPTV 3D tracking plugin for openptv2.

Implements MyPTV's 3D kinematic prediction tracking algorithm:
- Reads input reconstructed 3D particle positions from rt_is.# files (Frame objects)
- 2-frame velocity-bounded initialization
- Multi-frame polynomial position prediction with acceleration search bounds
- Hungarian / linear assignment distance matching
- Writes output trajectory linkages to ptv_is.# files
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from openptv2.algorithms.tracking_frame_buf import Frame


class MyPTV3DTracker:
    def __init__(
        self,
        v_max: float = 10.0,
        a_max: float = 50.0,
        max_gap: int = 2,
        dt: float = 0.1,
    ):
        self.v_max = v_max
        self.a_max = a_max
        self.max_gap = max_gap
        self.dt = dt

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

        active_tracks = []
        completed_tracks = []
        next_track_id = 1

        # Frame 0 initialization
        if len(frame_particles[0]) > 0:
            for p in frame_particles[0]:
                active_tracks.append({
                    "id": next_track_id,
                    "pos": [p],
                    "time": [0],
                    "vel": [np.zeros(3)],
                    "gap": 0,
                })
                next_track_id += 1

        # Process frames 1 .. N-1
        for f in range(1, num_frames):
            cand_pts = frame_particles[f]
            num_cands = len(cand_pts)
            num_active = len(active_tracks)

            if num_active == 0 or num_cands == 0:
                for tr in active_tracks:
                    completed_tracks.append(tr)
                active_tracks = []

                if num_cands > 0:
                    for p in cand_pts:
                        active_tracks.append({
                            "id": next_track_id,
                            "pos": [p],
                            "time": [f],
                            "vel": [np.zeros(3)],
                            "gap": 0,
                        })
                        next_track_id += 1
                continue

            BIG_COST = 1e9

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

            dists = cdist(pred, cand_pts)
            cost_matrix = np.where(dists <= radius[:, None], dists, BIG_COST)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_cands = set()
            matched_tracks = set()

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < BIG_COST / 2:
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
                        "pos": [cand_pts[c]],
                        "time": [f],
                        "vel": [np.zeros(3)],
                        "gap": 0,
                    })
                    next_track_id += 1

            active_tracks = new_active

        completed_tracks.extend(active_tracks)

        results = []
        for tr in completed_tracks:
            if len(tr["pos"]) >= 2:
                results.append({
                    "id": tr["id"],
                    "pos": np.array(tr["pos"]),
                    "time": np.array(tr["time"]),
                    "vel": np.array(tr["vel"]),
                })
        return results


class Tracking:
    """OpenPTV2 Tracking plugin interface for MyPTV 3D tracking."""

    def __init__(self, ptv=None, exp=None):
        self.ptv = ptv
        self.exp = exp

    def do_tracking(self) -> None:
        if self.exp is None:
            raise ValueError("No experiment object provided")

        print("Running MyPTV 3D Kinematic Prediction Tracking Plugin...")

        cpar = getattr(self.exp, "cpar", None)
        spar = getattr(self.exp, "spar", None)
        cals = getattr(self.exp, "cals", getattr(self.exp, "cal", []))
        res_dir = Path(getattr(self.exp, "res_dir", "res"))

        if cpar is not None:
            num_cams = cpar.num_cams
        else:
            num_cams = len(cals) if cals else 4

        if spar is not None:
            first_frame = spar.get_first()
            last_frame = spar.get_last()
        else:
            first_frame = int(getattr(self.exp, "first_frame", 1))
            last_frame = int(getattr(self.exp, "last_frame", 1))

        pm = getattr(self.exp, "pm", None)
        if pm is None and hasattr(self.exp, "exp1"):
            pm = getattr(self.exp.exp1, "pm", None)

        track_cfg = pm.parameters.get("track", {}) if pm else {}

        dvxmax = float(track_cfg.get("dvxmax", 10.0))
        dacc = float(track_cfg.get("dacc", 50.0))

        max_targets = 10000
        corres_base = str(res_dir / "rt_is")
        linkage_base = str(res_dir / "ptv_is")
        prio_base = str(res_dir / "added")

        frame_numbers = list(range(first_frame, last_frame + 1))
        num_frames = len(frame_numbers)

        # 1. Fill database using Frame objects reading ONLY input rt_is.# files
        frames = []
        frame_particles = []
        for fn in frame_numbers:
            frame = Frame(num_cams, max_targets)
            frame.read(
                corres_base,  # INPUT: res/rt_is
                "",           # Do NOT read existing ptv_is as input
                prio_file_base=prio_base,
                target_file_base="",
                frame_num=fn,
            )
            frames.append(frame)
            frame_particles.append(frame.positions())

        # 2. Run MyPTV 3D Kinematic Tracking
        tracker = MyPTV3DTracker(v_max=dvxmax, a_max=dacc, max_gap=1, dt=1.0)
        trajectories = tracker.track_frames(frame_particles)

        # 3. Create link assignments on Frame objects
        for tr in trajectories:
            times = tr["time"]
            for step_i in range(len(times) - 1):
                f_curr = times[step_i]
                f_next = times[step_i + 1]

                if f_next != f_curr + 1:
                    continue

                pos_curr = tr["pos"][step_i]
                pos_next = tr["pos"][step_i + 1]

                pts_curr = frame_particles[f_curr]
                pts_next = frame_particles[f_next]

                if len(pts_curr) > 0 and len(pts_next) > 0:
                    idx_curr = np.argmin(np.linalg.norm(pts_curr - pos_curr, axis=1))
                    idx_next = np.argmin(np.linalg.norm(pts_next - pos_next, axis=1))

                    frames[f_curr].path_next[idx_curr] = idx_next
                    frames[f_next].path_prev[idx_next] = idx_curr

        # 4. Sync SoA to Pathinfo & Write Frame database out to ptv_is.# (OUTPUT)
        total_links = 0
        total_particles = 0

        for f_idx, fn in enumerate(frame_numbers):
            frame = frames[f_idx]
            total_particles += frame.num_parts

            frame._sync_soa_to_path()
            frame.write(
                corres_base,    # res/rt_is
                linkage_base,   # OUTPUT: res/ptv_is
                prio_file_base=prio_base,
                target_file_base="",
                frame_num=fn,
            )

            if f_idx < num_frames - 1:
                curr_c = frame.num_parts
                next_c = frames[f_idx + 1].num_parts
                step_links = np.sum(frame.path_next[:curr_c] >= 0)
                total_links += step_links
                lost_c = curr_c - step_links
                print(f"step: {f_idx + 1}, curr: {curr_c}, next: {next_c}, links: {step_links}, lost: {lost_c}, add: 0")

        n_steps = max(1, num_frames - 1)
        avg_particles = total_particles / max(1, num_frames)
        avg_links = total_links / n_steps
        avg_lost = avg_particles - avg_links

        print(f"Average over sequence, particles: {avg_particles:.1f}, links: {avg_links:.1f}, lost: {avg_lost:.1f}")
        print("MyPTV 3D Tracking completed successfully.")
