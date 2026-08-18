"""proPTV-style probabilistic tracking plugin for openptv2.

This plugin borrows the central *concept* of proPTV (Barta et al., Meas. Sci.
Technol. 35 (2024) 105302) — modelling each track's time-position history with
a set of smooth basis functions (Gaussian Mixture Model / Savitzky-Golay) so
that velocity and acceleration at the current frame are estimated robustly and
the next position is predicted with a physically smooth extrapolation — but
implements it on openptv2's own trackers/assignment machinery, tracking in 3D
straight from the already-triangulated ``rt_is.#`` particles.

This avoids porting proPTV's full 2D-image triangulation pipeline.  We reuse:

  * GMM / SavGol smoothing for velocity+acceleration estimation
  * ``openptv2.plugins._assignment.match_within_radius`` for the
    radius-limited Hungarian assignment
  * ``openptv2.tracking_cost.compute_multi_term_cost_matrix`` for a
    distance + velocity + acceleration continuity cost

Reference: Barta et al., "proPTV — a probability-based particle tracking
velocimetry framework", Meas. Sci. Technol. 35, 105302 (2024).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from openptv2.algorithms.tracking_frame_buf import Frame
from openptv2.plugins._assignment import match_within_radius
from openptv2.plugins.proptv import ProPTVConfig
from openptv2.plugins.proptv.initialisation import (
    init_acceleration_3d,
    init_velocity_3d,
)
from openptv2.plugins.proptv.prediction import GMM, Approximate
from openptv2.tracking_cost import (
    CostWeights,
    compute_multi_term_cost_matrix,
)

# Minimum track length (inclusive) before the GMM predictor is reliable.
_MIN_GMM_HISTORY = 3


def _fit_gmm_smooth(times, pos):
    """Return (X, V, A) at the last time step via GMM basis approximation.

    Falls back to a finite-difference / constant-velocity estimate if the GMM
    solve fails or the history is too short.
    """
    n = len(pos)
    if n < _MIN_GMM_HISTORY:
        if n == 1:
            return pos[-1], np.zeros(3), np.zeros(3)
        v = pos[-1] - pos[-2]
        return pos[-1], v, np.zeros(3)
    try:
        w, psi_X, psi_V, psi_A = GMM(np.asarray(times), np.asarray(pos))
        X, V, A = Approximate(np.asarray(times), w, psi_X, psi_V, psi_A)
        return X[-1], V[-1], A[-1]
    except (np.linalg.LinAlgError, ValueError):
        v = pos[-1] - pos[-2]
        acc = (pos[-1] - 2 * pos[-2] + pos[-3]) if n >= 3 else np.zeros(3)
        return pos[-1], v, acc


def _smooth_history(pos):
    """Re-fit the whole track's velocities/accelerations with SavGol."""
    return (
        init_velocity_3d(np.asarray(pos)),
        init_acceleration_3d(np.asarray(pos)),
    )


class ProPTVTracker:
    """proPTV-concept 3D tracker (GMM prediction + Hungarian assignment)."""

    def __init__(self, config: ProPTVConfig | None = None):
        self.config = config or ProPTVConfig()

    # ------------------------------------------------------------------
    def track_frames(self, frame_particles: list[np.ndarray]) -> list[dict]:
        """Track a list of per-frame 3D particle arrays (N_i, 3).

        Returns a list of trajectory dicts with keys
        ``id``, ``time``, ``pos``, ``vel``, ``acc``.
        """
        cfg = self.config
        num_frames = len(frame_particles)
        if num_frames < 2:
            return []

        active: list[dict] = []
        completed: list[dict] = []
        next_track_id = 1

        # ── Initialisation: link the first t_init frames █────────────
        active, next_track_id = self._initialise(
            frame_particles, cfg, next_track_id
        )

        # ── Main loop █───────────────────────────────────────────────
        for f in range(cfg.t_init, num_frames):
            cand = frame_particles[f]
            if len(cand) == 0:
                completed.extend(active)
                active = [self._new_track(next_track_id + i, cand, f)
                          for i in range(len(cand))]
                next_track_id += len(cand)
                continue

            active, completed, next_track_id = self._advance_frame(
                f, cand, active, completed, next_track_id, cfg
            )

        completed.extend(active)
        return self._finalize(completed)

    # ------------------------------------------------------------------
    def _new_track(self, track_id, pos, frame_idx):
        return {
            "id": track_id,
            "time": [frame_idx],
            "pos": [pos.copy()],
            "vel": [np.zeros(3)],
            "acc": [np.zeros(3)],
            "gap": 0,
        }

    def _initialise(self, frames, cfg, next_track_id):
        """Seed tracks over the first t_init frames via NN linking."""
        active = []
        t_init = min(cfg.t_init, len(frames))
        if t_init < 2 or len(frames[0]) == 0:
            # not enough frames: start one track per particle in frame 0
            for p in frames[0]:
                active.append(self._new_track(next_track_id, p, 0))
                next_track_id += 1
            return active, next_track_id

        # Link frame i -> i+1 for the first t_init frames.
        # Start a candidate from every particle in frame 0, extend greedily.
        paths = {i: [] for i in range(t_init)}
        for i, p in enumerate(frames[0]):
            paths[0].append((i, p))
        used = [np.zeros(len(frames[i]), dtype=bool) for i in range(t_init)]

        # greedy per-seed forward linking (first-neighbour)
        for seed_i, p0 in enumerate(frames[0]):
            pts = [p0]
            idx = [seed_i]
            ok = True
            for f in range(1, t_init):
                prev_p = pts[-1]
                if len(frames[f]) == 0:
                    ok = False
                    break
                d = np.linalg.norm(frames[f] - prev_p, axis=1)
                j = int(np.argmin(d))
                step = d[j]
                if step > cfg.maxvel:
                    ok = False
                    break
                pts.append(frames[f][j])
                idx.append(j)
            if not ok or len(pts) < 2:
                continue
            # angle check
            if not self._angle_ok(np.array(pts), cfg.angle):
                continue
            tr = {
                "id": next_track_id,
                "time": list(range(t_init)),
                "pos": [p.copy() for p in pts],
                "gap": 0,
            }
            vel, acc = _smooth_history(np.array(pts))
            tr["vel"] = [v.copy() for v in vel]
            tr["acc"] = [a.copy() for a in acc]
            active.append(tr)
            next_track_id += 1
            # mark used particles so seeds don't reuse them
            for f in range(t_init):
                used[f][idx[f]] = True

        # remaining frame-0 particles become single-point tracks
        for i, p in enumerate(frames[0]):
            if not used[0][i]:
                active.append(self._new_track(next_track_id, p, 0))
                next_track_id += 1

        return active, next_track_id

    @staticmethod
    def _angle_ok(pts, max_angle_deg):
        if len(pts) < 3:
            return True
        vels = np.diff(pts, axis=0)
        for k in range(len(vels) - 1):
            v1, v2 = vels[k], vels[k + 1]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 1e-9 and n2 > 1e-9:
                cosa = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
                if np.degrees(np.arccos(cosa)) > max_angle_deg:
                    return False
        return True

    # ------------------------------------------------------------------
    def _advance_frame(self, f, cand, active, completed, next_track_id, cfg):
        num_active = len(active)
        if num_active == 0:
            for p in cand:
                active.append(self._new_track(next_track_id, p, f))
                next_track_id += 1
            return active, completed, next_track_id

        # Predict next position for each active track via GMM.
        last_p = np.array([tr["pos"][-1] for tr in active])
        pred = np.array([last_p[i] for i in range(num_active)])
        seeded = np.zeros(num_active, dtype=bool)
        for i, tr in enumerate(active):
            if len(tr["pos"]) >= _MIN_GMM_HISTORY:
                try:
                    P, V, A = _fit_gmm_smooth(tr["time"], tr["pos"])
                except Exception:
                    P = tr["pos"][-1]
                pred[i] = P
                seeded[i] = True

        # Per-track search radius: wider for single-point (uses v_max), else
        # the GMM-predicted uncertainty scaled by maxvel.
        radius = np.where(seeded, cfg.maxvel, cfg.maxvel)

        # Cost matrix: distance + velocity + acceleration continuity.
        n_pred = num_active
        n_cand = len(cand)
        last_v = np.array([tr["vel"][-1] for tr in active])
        last_acc = np.array([tr["acc"][-1] for tr in active])
        weights = CostWeights(w_distance=1.0, w_velocity=0.6, w_acceleration=0.3)
        cost = compute_multi_term_cost_matrix(
            pred_pos=pred,
            cand_pos=cand,
            pred_vel=last_v,
            pred_acc=last_acc,
            weights=weights,
            dt=1.0,
        )

        row_ind, col_ind = match_within_radius(pred, cand, radius, cost_matrix=cost)

        matched_cands = set()
        matched_tracks = set()
        for r, c in zip(row_ind, col_ind):
            tr = active[r]
            new_p = cand[c].copy()
            dt_eff = (f - tr["time"][-1]) * max(1, cfg.dt)
            tr["pos"].append(new_p)
            tr["time"].append(f)
            tr["gap"] = 0
            # recompute smoothed vel/acc over full history (proPTV concept)
            vel, acc = _smooth_history(np.array(tr["pos"]))
            tr["vel"] = [v.copy() for v in vel]
            tr["acc"] = [a.copy() for a in acc]
            matched_tracks.add(r)
            matched_cands.add(c)

        new_active = []
        for i, tr in enumerate(active):
            if i in matched_tracks:
                new_active.append(tr)
                continue
            tr["gap"] += 1
            if tr["gap"] <= (1 if cfg.gaptracking else 0):
                new_active.append(tr)
            else:
                completed.append(tr)

        for c in range(n_cand):
            if c not in matched_cands:
                new_active.append(self._new_track(next_track_id, cand[c], f))
                next_track_id += 1

        return new_active, completed, next_track_id

    # ------------------------------------------------------------------
    @staticmethod
    def _finalize(completed):
        out = []
        for tr in completed:
            if len(tr["pos"]) >= 2:
                out.append(
                    {
                        "id": tr["id"],
                        "time": np.array(tr["time"]),
                        "pos": np.array(tr["pos"]),
                        "vel": np.array(tr["vel"]),
                        "acc": np.array(tr["acc"]),
                    }
                )
        return out


class Tracking:
    """OpenPTV2 Tracking plugin interface for proPTV-concept tracking."""

    def __init__(self, ptv=None, exp=None):
        self.ptv = ptv
        self.exp = exp

    def do_tracking(self) -> None:
        if self.exp is None:
            raise ValueError("No experiment object provided")

        print("Running proPTV-concept (GMM 3D) Tracking Plugin...")

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
        proptv_cfg = pm.parameters.get("proptv", {}) if pm else {}

        from openptv2.tracking_presets import unified_angle_deg, unified_velocity_bound

        # maxvel/angle default from the SAME unified per-axis dv box / gon
        # angle every other tracker reads (unified_velocity_bound,
        # unified_angle_deg -- angle in particular needs the gon->degrees
        # conversion, since proptv's own angle concept, "max angle between
        # successive velocity vectors", is in degrees while track.angle is
        # trackcorr's gon convention). proptv.maxvel/proptv.angle remain a
        # deliberate per-tracker override for when the unified default
        # genuinely doesn't fit this engine.
        maxvel = float(proptv_cfg.get("maxvel", unified_velocity_bound(track_cfg)))
        angle = float(proptv_cfg.get("angle", unified_angle_deg(track_cfg, default_deg=30.0)))
        t_init = int(proptv_cfg.get("t_init", 4))
        gaptracking = bool(proptv_cfg.get("gaptracking", False))

        cfg = ProPTVConfig(
            t_init=t_init,
            maxvel=maxvel,
            angle=angle,
            activeMatches_extend=int(proptv_cfg.get("activeMatches_extend", 3)),
            backtracking=bool(proptv_cfg.get("backtracking", False)),
            gaptracking=gaptracking,
        )

        from openptv2.gui.ptv import _open_run_store

        store = _open_run_store(self.exp)

        max_targets = 10000
        corres_base = str(res_dir / "rt_is")
        linkage_base = str(res_dir / "ptv_is")
        prio_base = str(res_dir / "added")

        frame_numbers = list(range(first_frame, last_frame + 1))
        num_frames = len(frame_numbers)

        # 1. Read 3D particles per frame -- store first, ascii rt_is fallback.
        frames = []
        frame_particles = []
        for fn in frame_numbers:
            frame = Frame(num_cams, max_targets)
            frame.read(
                corres_base, "", prio_file_base=prio_base,
                target_file_base="", frame_num=fn, store=store,
            )
            frames.append(frame)
            frame_particles.append(frame.positions())

        # 2. Run proPTV-concept 3D tracking.
        tracker = ProPTVTracker(config=cfg)
        trajectories = tracker.track_frames(frame_particles)

        # 3. Backward pass if requested (greedy reverse re-link).
        if cfg.backtracking:
            trajectories = self._backtrack(
                trajectories, frame_particles, maxvel
            )

        # 4. Write linkages to ptv_is.  `tr["time"]` holds 0-based frame
        # indices (relative to the sequence start), so `idx = time` maps
        # directly into `frame_particles` / `frames`.
        for tr in trajectories:
            times = tr["time"]
            positions = tr["pos"]
            for step_i in range(len(times) - 1):
                f_curr = int(times[step_i])
                f_next = int(times[step_i + 1])
                if f_next != f_curr + 1:
                    continue
                if f_curr < 0 or f_next >= num_frames:
                    continue
                pts_curr = frame_particles[f_curr]
                pts_next = frame_particles[f_next]
                if len(pts_curr) == 0 or len(pts_next) == 0:
                    continue
                i_curr = int(np.argmin(np.linalg.norm(pts_curr - positions[step_i], axis=1)))
                i_next = int(np.argmin(np.linalg.norm(pts_next - positions[step_i + 1], axis=1)))
                frames[f_curr].path_next[i_curr] = i_next
                frames[f_next].path_prev[i_next] = i_curr

        # 5. Write output.
        total_links = 0
        total_particles = 0
        for f_idx, fn in enumerate(frame_numbers):
            frame = frames[f_idx]
            total_particles += frame.num_parts
            frame._sync_soa_to_path()
            frame.write(
                corres_base, linkage_base,
                prio_file_base=prio_base, target_file_base=None, frame_num=fn,
                store=store,
            )
            if f_idx < num_frames - 1:
                links = int(np.sum(frame.path_next[: frame.num_parts] >= 0))
                total_links += links
                print(f"  Frame {fn}: {frame.num_parts} particles, {links} links")

        avg_links = total_links / max(1, num_frames - 1)
        print(
            f"proPTV-concept Tracking completed: {total_links} total links, "
            f"avg {avg_links:.1f}/step"
        )

    @staticmethod
    def _backtrack(completed, frame_particles, maxvel):
        """Greedy reverse pass: extend short tracks backward one frame."""
        # Rebuild a map frame -> particle -> track-id occupancy to avoid
        # reassigning an already-used point.
        used = [set() for _ in frame_particles]
        for tr in completed:
            for f, p in zip(tr["time"], tr["pos"]):
                if 0 <= f < len(used) and len(frame_particles[f]) > 0:
                    i = int(np.argmin(np.linalg.norm(frame_particles[f] - p, axis=1)))
                    used[f].add(i)

        extended = 0
        for tr in completed:
            if len(tr["time"]) < 2 or tr["time"][0] <= 0:
                continue
            f0 = tr["time"][0]
            prev_frame = frame_particles[f0 - 1]
            if len(prev_frame) == 0:
                continue
            p0 = tr["pos"][0]
            d = np.linalg.norm(prev_frame - p0, axis=1)
            j = int(np.argmin(d))
            if d[j] <= maxvel and j not in used[f0 - 1]:
                used[f0 - 1].add(j)
                tr["time"].insert(0, f0 - 1)
                tr["pos"].insert(0, prev_frame[j].copy())
                vel, acc = _smooth_history(np.array(tr["pos"]))
                tr["vel"] = [v.copy() for v in vel]
                tr["acc"] = [a.copy() for a in acc]
                extended += 1
        print(f"  Backtrack: extended {extended} tracks by one frame")
        return completed
