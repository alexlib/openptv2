"""fast_3d_smooth — 3D-only tracking with a Savitzky-Golay velocity model.

Same contract as fast_3d (reads ``rt_is.#``, writes ``ptv_is.#``, no 2D
epipolar search, no new-particle seeding) but the prediction replaces
fast_3d's 2-point extrapolation ``2*curr - prev`` with a Savitzky-Golay
smoothed velocity estimate over a short track history.

Why this helps
--------------
fast_3d's bare linear extrapolation carries the full position noise of the
last two frames (velocity noise ~ sqrt(2)*sigma_pos).  Smoothing the last
``window`` positions with a poly order-3 SG filter cuts that velocity noise
roughly in half at window ~5 while preserving constant-acceleration motion
(SG-3 is exact on quadratics).  Assignment then happens through the same
radius-limited Hungarian used by myptv/proptv
(``openptv2.plugins._assignment.match_within_radius``), which resolves
order-dependent greedy errors that fragment fast_3d at crossing particles.

Parameters (read from the YAML ``track`` section)
-------------------------------------------------
dvxmin/dvxmax/dvymin/dvymax/dvzmin/dvzmax : search window for a track with
    no velocity estimate (mm/frame).
dacc                                : search radius around the predicted
    position for a track with velocity history (mm).
(dataset-derived from a myptv probe: dv* ~ +/-6, dacc ~ 3-4.)
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

import numpy as np

from openptv2.algorithms.tracking_frame_buf import Frame
from openptv2.plugins._assignment import match_within_radius
from openptv2.tracking_cost import CostWeights, compute_multi_term_cost_matrix

log = logging.getLogger("openptv2.fast_3d_smooth")

_MIN_HISTORY = 2  # frames before a velocity estimate exists
_SG_MAX_WINDOW = 5
_SG_POLY = 3


@functools.lru_cache(maxsize=8)
def _sg_deriv_coeffs(window: int, poly: int) -> np.ndarray:
    """SG derivative coefficients reproducing scipy's interp-mode edge fit.

    The reference implementation reads ``savgol_filter(y, w, p, deriv=1,
    mode='interp')[-1]``; that trailing-edge value is algebraically equal to
    ``-dot(y[-w:], savgol_coeffs(w, p, deriv=1, pos=0))`` (verified to ~1e-14
    on random walks).  This caches the negated kernel so the whole batch can
    be evaluated with one ``tensordot`` and no per-sample scipy calls.
    """
    from scipy.signal import savgol_coeffs

    return -savgol_coeffs(window, poly, deriv=1, pos=0)


def _estimate_velocities(hist: list[np.ndarray], window: int = _SG_MAX_WINDOW) -> np.ndarray:
    """SG-smoothed velocity at the latest sample, vectorised across tracks.

    ``hist[i]`` is the (m_i, 3) history of track i, newest sample last.
    Every track is assigned its effective window ``w`` (= odd(min(m, window)))
    and all tracks sharing the same window are processed in a single
    vectorised Savitzky-Golay derivative via a cached coefficient vector.
    Returns a (n_tracks, 3) array of velocities (mm/frame).
    """
    n = len(hist)
    if n == 0:
        return np.zeros((0, 3))
    vel = np.empty((n, 3))

    key = []
    for track in hist:
        m = len(track)
        if m >= 3:
            w = min(window, m)
            if w % 2 == 0:
                w -= 1
            if w >= 3:
                key.append((w, min(_SG_POLY, w - 1)))
                continue
        key.append((0, 0))  # two-point fallback bucket

    groups: dict[tuple[int, int], list[int]] = {}
    for i, wp in enumerate(key):
        groups.setdefault(wp, []).append(i)

    for (w, p), idxs in groups.items():
        if w == 0:
            for i in idxs:
                h = hist[i]
                if len(h) == 1:
                    vel[i] = np.zeros(3)
                else:
                    vel[i] = h[-1] - h[-2]
            continue
        coeffs = _sg_deriv_coeffs(w, p)
        tail = np.empty((len(idxs), w, 3))
        for j, i in enumerate(idxs):
            tail[j] = hist[i][-w:]
        vel[idxs] = np.tensordot(tail, coeffs, axes=([1], [0]))
    return vel


class Fast3DSmoothTracker:
    """Position-space tracker with SG-smoothed velocity prediction.

    Per frame, each active track is extrapolated with its current smoothed
    velocity (constant-velocity within the SG window), then the candidates
    are matched with a radius-limited Hungarian assignment per connected
    component.  Unmatched candidates seed new tracks; unmatched tracks may
    bridge up to ``max_gap`` gaps.
    """

    def __init__(
        self,
        v_max: float = 10.0,
        dacc: float = 3.0,
        max_gap: int = 1,
        dt: float = 1.0,
        smooth_window: int = _SG_MAX_WINDOW,
        weights: CostWeights | None = None,
    ):
        self.v_max = v_max
        self.dacc = dacc
        self.max_gap = max_gap
        self.dt = dt
        self.smooth_window = smooth_window
        self.weights = weights

    # ------------------------------------------------------------------
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
        tracks = []
        for p in cand_pts:
            tracks.append(self._new_track(next_track_id, p, frame_idx))
            next_track_id += 1
        return tracks, next_track_id

    def _advance_frame(self, f, cand_pts, active_tracks, completed_tracks, next_track_id):
        num_cands = len(cand_pts)
        if len(active_tracks) == 0 or num_cands == 0:
            completed_tracks.extend(active_tracks)
            new_active, next_track_id = self._seed_tracks(cand_pts, f, next_track_id)
            return new_active, completed_tracks, next_track_id

        # ---- prediction + radius for every active track at once ----
        hist = [tr["pos"] for tr in active_tracks]
        last_p = np.array([h[-1] for h in hist])
        has_vel = np.array([len(h) >= _MIN_HISTORY for h in hist])
        last_vel = np.where(has_vel[:, None], _estimate_velocities(hist, self.smooth_window), 0.0)
        seeded = has_vel

        # extrapolate one step with the smoothed velocity
        pred = last_p + last_vel * self.dt
        radius = np.where(seeded, self.dacc, self.v_max)

        if self.weights is not None:
            cost_mat = compute_multi_term_cost_matrix(
                pred_pos=pred,
                cand_pos=cand_pts,
                pred_vel=last_vel,
                weights=self.weights,
                dt=self.dt,
            )
        else:
            cost_mat = None
        row_ind, col_ind = match_within_radius(pred, cand_pts, radius, cost_matrix=cost_mat)

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
    def _finalize(completed_tracks):
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

    def track_frames(self, frame_particles):
        num_frames = len(frame_particles)
        if num_frames < 2:
            return []
        active = []
        completed = []
        next_track_id = 1
        for f, cand in enumerate(frame_particles):
            active, completed, next_track_id = self._advance_frame(
                f, cand, active, completed, next_track_id
            )
        completed.extend(active)
        return self._finalize(completed)


class Tracking:
    """OpenPTV tracking plugin entry: reads rt_is, runs fast_3d_smooth,
    writes ptv_is linkage files (same I/O contract as fast_3d/myptv)."""

    def __init__(self, ptv=None, exp=None):
        self.ptv = ptv
        self.exp = exp

    def do_tracking(self) -> None:
        if self.exp is None:
            raise ValueError("No experiment object provided")

        print("Running Fast 3D Smooth (Savitzky-Golay velocity) Tracking...")

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
        smooth_cfg = pm.parameters.get("fast_3d_smooth", {}) if pm else {}

        dvxmax = float(track_cfg.get("dvxmax", 10.0))
        dacc = float(track_cfg.get("dacc", 3.0))
        max_gap = int(smooth_cfg.get("max_gap", 1))
        smooth_window = int(smooth_cfg.get("smooth_window", _SG_MAX_WINDOW))

        max_targets = 10000
        corres_base = str(res_dir / "rt_is")
        linkage_base = str(res_dir / "ptv_is")
        prio_base = str(res_dir / "added")

        frame_numbers = list(range(first_frame, last_frame + 1))
        num_frames = len(frame_numbers)

        # 1. Fill database from res/rt_is only (positions).
        frames = []
        frame_particles = []
        for fn in frame_numbers:
            frame = Frame(num_cams, max_targets)
            frame.read(
                corres_base,
                "",
                prio_file_base=prio_base,
                target_file_base="",
                frame_num=fn,
            )
            frames.append(frame)
            frame_particles.append(frame.positions())

        # 2. Run the smoothed tracker.
        tracker = Fast3DSmoothTracker(
            v_max=dvxmax, dacc=dacc, max_gap=max_gap,
            smooth_window=smooth_window, dt=1.0,
        )
        trajectories = tracker.track_frames(frame_particles)

        # 3. Populate Frame objects with link assignments.
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
                if len(pts_curr) and len(pts_next):
                    idx_curr = int(np.argmin(np.linalg.norm(pts_curr - pos_curr, axis=1)))
                    idx_next = int(np.argmin(np.linalg.norm(pts_next - pos_next, axis=1)))
                    frames[f_curr].path_next[idx_curr] = idx_next
                    frames[f_next].path_prev[idx_next] = idx_curr

        # 4. Sync + write the frame database out to res/ptv_is.#
        total_links = 0
        total_particles = 0
        for f_idx, fn in enumerate(frame_numbers):
            frame = frames[f_idx]
            total_particles += frame.num_parts
            frame._sync_soa_to_path()
            frame.write(
                corres_base,
                linkage_base,
                prio_file_base=prio_base,
                target_file_base=None,
                frame_num=fn,
            )
            if f_idx < num_frames - 1:
                curr_c = frame.num_parts
                next_c = frames[f_idx + 1].num_parts
                step_links = int(np.sum(frame.path_next[:curr_c] >= 0))
                total_links += step_links
                lost_c = curr_c - step_links
                print(
                    f"step: {f_idx + 1}, curr: {curr_c}, next: {next_c}, "
                    f"links: {step_links}, lost: {lost_c}, add: 0"
                )

        n_steps = max(1, num_frames - 1)
        avg_particles = total_particles / max(1, num_frames)
        avg_links = total_links / n_steps
        avg_lost = avg_particles - avg_links
        print(
            f"Average over sequence, particles: {avg_particles:.1f}, "
            f"links: {avg_links:.1f}, lost: {avg_lost:.1f}"
        )
        print("Fast 3D Smooth Tracking completed successfully.")
