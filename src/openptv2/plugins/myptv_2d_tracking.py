"""MyPTV 2D tracking plugin for openptv2.

Implements MyPTV's 2D image-space tracking algorithm per camera:
- Reads input 3D particle correspondences from rt_is.# files (Frame objects)
- Projects 3D positions to 2D image coordinates per camera
- 2D pixel displacement search bounds & linear assignment tracking per camera
- Multi-camera consensus link aggregation updated directly in Frame database
- Writes output trajectory linkages to ptv_is.# files
"""

from __future__ import annotations
from pathlib import Path
import numpy as np

from openptv2.algorithms.tracking_frame_buf import Frame
from openptv2.algorithms.imgcoord import img_coord_batch
from openptv2.plugins._assignment import match_within_radius


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
            List of 2D trajectories containing 'pos_2d', 'indices', 'time', and 'id'.
        """
        num_frames = len(frame_blobs)
        if num_frames < 2:
            return []

        active_tracks = []
        completed_tracks = []
        next_track_id = 1

        if len(frame_blobs[0]) > 0:
            for idx, p in enumerate(frame_blobs[0]):
                active_tracks.append({
                    "id": next_track_id,
                    "indices": [idx],
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
                    for idx, p in enumerate(cands):
                        active_tracks.append({
                            "id": next_track_id,
                            "indices": [idx],
                            "pos_2d": [p],
                            "time": [f],
                            "vel_2d": [np.zeros(2)],
                            "gap": 0,
                        })
                        next_track_id += 1
                continue

            # Prediction for every active track at once: a single-point track
            # has no velocity estimate yet and predicts "no motion", a seeded
            # track extrapolates its last pixel displacement.
            last_p = np.array([tr["pos_2d"][-1] for tr in active_tracks])
            last_v = np.array([tr["vel_2d"][-1] for tr in active_tracks])
            seeded = np.fromiter(
                (len(tr["pos_2d"]) > 1 for tr in active_tracks),
                dtype=bool,
                count=num_active,
            )
            pred = np.where(seeded[:, None], last_p + last_v, last_p)

            row_ind, col_ind = match_within_radius(pred, cands, self.max_pixel_disp)

            matched_cands = set()
            matched_tracks = set()

            for r, c in zip(row_ind, col_ind):
                tr = active_tracks[r]
                new_p = cands[c]
                dt_eff = f - tr["time"][-1]
                v_new = (new_p - tr["pos_2d"][-1]) / max(dt_eff, 1)

                tr["indices"].append(c)
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
                        "indices": [c],
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
                    "indices": tr["indices"],
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

        cpar = getattr(self.exp, "cpar", None)
        spar = getattr(self.exp, "spar", None)
        cals = getattr(self.exp, "cals", getattr(self.exp, "cal", []))
        res_dir = Path(getattr(self.exp, "res_dir", "res"))

        if cpar is not None:
            num_cams = cpar.num_cams
            mm = cpar.mm
        else:
            num_cams = len(cals) if cals else 4
            from openptv2.parameters import ControlParams
            mm = ControlParams().mm

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

        dvmx = float(track_cfg.get("dvmx", 20.0))
        if dvmx < 5.0:
            dvmx = 20.0

        max_targets = 10000
        corres_base = str(res_dir / "rt_is")
        linkage_base = str(res_dir / "ptv_is")
        prio_base = str(res_dir / "added")

        frame_numbers = list(range(first_frame, last_frame + 1))
        num_frames = len(frame_numbers)

        # 1. Fill database using Frame objects reading ONLY input rt_is.# files
        frames = []
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

        # 2. Extract 2D projected points for each camera across frames
        cams_2d_blobs = []
        for cam_idx in range(num_cams):
            cal = cals[cam_idx] if cam_idx < len(cals) else None

            frame_blobs_cam = []
            for f_idx in range(num_frames):
                frame = frames[f_idx]
                pts_3d = frame.positions()
                if len(pts_3d) > 0 and cal is not None:
                    pix = img_coord_batch(pts_3d, cal, mm)
                else:
                    pix = np.empty((0, 2))
                frame_blobs_cam.append(pix)

            cams_2d_blobs.append(frame_blobs_cam)

        # 3. Track 2D points per camera & tally multi-camera consensus votes
        vote_matrices = [
            np.zeros((frames[f].num_parts, frames[f + 1].num_parts), dtype=np.int32)
            for f in range(num_frames - 1)
        ]

        tracker_2d = MyPTV2DTracker(max_pixel_disp=dvmx, max_gap=1)

        for cam_idx in range(num_cams):
            frame_blobs_cam = cams_2d_blobs[cam_idx]
            trajectories_2d = tracker_2d.track_2d_blobs(frame_blobs_cam)

            for tr in trajectories_2d:
                times = tr["time"]
                indices = tr["indices"]
                for step_i in range(len(times) - 1):
                    f_curr = times[step_i]
                    f_next = times[step_i + 1]

                    if f_next != f_curr + 1:
                        continue

                    idx_curr = indices[step_i]
                    idx_next = indices[step_i + 1]

                    if (
                        idx_curr < frames[f_curr].num_parts
                        and idx_next < frames[f_next].num_parts
                    ):
                        vote_matrices[f_curr][idx_curr, idx_next] += 1

        # 4. Perform Bipartite Matching on consensus votes & update Frame link arrays
        for f_idx in range(num_frames - 1):
            vm = vote_matrices[f_idx]
            frame_curr = frames[f_idx]
            frame_next = frames[f_idx + 1]

            if vm.size > 0 and np.max(vm) > 0:
                cost_matrix = 1000 - vm.astype(np.float64)
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                for r, c in zip(row_ind, col_ind):
                    if vm[r, c] >= 1:
                        frame_curr.path_next[r] = c
                        frame_next.path_prev[c] = r

        # 5. Sync SoA to Pathinfo & Write Frame database out to ptv_is.# (OUTPUT)
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
        print("MyPTV 2D Tracking completed successfully.")
