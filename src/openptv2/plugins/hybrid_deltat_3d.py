"""Hybrid multi-Δt 3D tracking plugin (Cierpka-style coarse-to-fine linking).

For slow flows sampled at high frame rates, per-frame displacement can be
smaller than the 3D reconstruction noise (poorly-conditioned tracking --
see openptv2.tracking_feasibility). No gate or assignment algorithm can
recover links the noise has erased at Δt=1, but over a larger step Δt=N
the displacement grows N-fold while the noise floor stays put.

This plugin exploits that in two passes:

1. **Coarse pass** — link only every N-th frame's particle cloud with the
   predictive Hungarian tracker (MyPTV3DTracker, which is already
   dt-correct via per-point timestamps). Search radii are scaled to the
   stride: an unseeded track searches a v_max*N ball; a seeded track
   allows for velocity underprediction plus acceleration curvature,
   ~ (N-1)*v_max + a_max*N²/2.
2. **Refine pass** — walk the intermediate frames of each coarse segment,
   predict the particle's position with cubic-Hermite interpolation of the
   segment endpoints' positions and velocities, and attach the nearest
   not-yet-used detection within ``refine_gate_mm``. Attached detections
   chain the endpoints through consecutive frames; where no detection fits
   the prediction the chain breaks into sub-chains (downstream stitching
   can re-glue those small gaps).

The default/fast trackers are untouched: this is an opt-in preset
(``track.preset: hybrid_deltat_3d``, or ``plugins.selected_tracking:
hybrid_deltat_3d``) for datasets whose conditioning report says individual
trajectories cannot be trusted at full frame rate.

Output linkage is written through the same Frame/rt_is/ptv_is/store path
as every other Python tracker, so post-processing (flowtracks, ensemble
binning) needs no changes. Velocities derived downstream are correct
across attached gaps because positions carry their true frame numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from openptv2.algorithms.tracking_frame_buf import Frame
from openptv2.plugins.myptv_3d_tracking import MyPTV3DTracker


def hybrid_track(
    frame_particles: list[np.ndarray],
    stride: int = 5,
    v_max: float = 10.0,
    a_max: float = 50.0,
    max_angle_deg: float | None = None,
    refine_gate: float = 0.8,
) -> list[dict]:
    """Coarse-Δt link then fine-refine a sequence of 3D particle clouds.

    Parameters
    ----------
    frame_particles : list of (N_i, 3) arrays, one per frame.
    stride : int
        Coarse-pass step N (link frames i and i+N first).
    v_max, a_max : float
        Per-frame velocity bound and acceleration search radius, same
        meaning as the standard trackers (mm/frame, mm/frame²).
    max_angle_deg : float, optional
        Cone-of-continuity filter forwarded to the coarse tracker.
    refine_gate : float
        Maximum distance (mm) between a Hermite-predicted intermediate
        position and a detection for the detection to be attached.

    Returns
    -------
    chains : list of dict
        Each chain is {"frame": list[int], "idx": list[int], "pos": ndarray}
        with strictly consecutive frames -- ready to be written as
        prev/next linkage. Detections belong to at most one chain.
    """
    n_frames = len(frame_particles)
    if n_frames < 2:
        return []
    stride = max(1, int(min(stride, n_frames - 1)))

    used = [np.zeros(len(fp), dtype=bool) for fp in frame_particles]

    # ---- coarse pass --------------------------------------------------
    # Always include the LAST frame so the sequence tail is covered too.
    coarse_idx = sorted(set(range(0, n_frames, stride)) | {n_frames - 1})
    coarse_frames = [frame_particles[i] for i in coarse_idx]
    # Radius budget for seeded tracks: MyPTV3DTracker predicts p + v_native
    # one COARSE step ahead, so the radius must absorb the remaining
    # (N-1)*v of constant-velocity travel plus a*N^2/2 of curvature.
    seed_radius = (stride - 1) * v_max + 0.5 * a_max * stride**2
    coarse_tracker = MyPTV3DTracker(
        v_max=v_max * stride,
        a_max=max(seed_radius, a_max),
        max_gap=0,
        dt=float(stride),
        max_angle_deg=max_angle_deg,
    )
    segments = coarse_tracker.track_frames(coarse_frames)

    # Recompute per-node velocities from TRUE frame gaps. The coarse
    # tracker divides by its uniform dt, which is wrong on any irregular
    # interval (e.g. the shortened final step to the last frame).
    for tr in segments:
        p = np.asarray(tr["pos"])
        t_real = np.asarray(coarse_idx, dtype=float)[np.asarray(tr["time"], dtype=int)]
        if len(p) < 2:
            continue
        dt = np.diff(t_real)
        v = np.zeros_like(p)
        v[:-1] = np.diff(p, axis=0) / dt[:, None]
        v[-1] = v[-2]
        tr["vel"] = v

    def claim(frame_li: int, pos: np.ndarray) -> int:
        """Reserve the nearest UNUSED cloud point to pos; -1 if none/close."""
        pts = frame_particles[frame_li]
        if len(pts) == 0:
            return -1
        d = np.linalg.norm(pts - np.asarray(pos), axis=1)
        d_used = np.where(used[frame_li], np.inf, d)
        k = int(np.argmin(d_used))
        return k if np.isfinite(d_used[k]) else -1

    # ---- refine pass --------------------------------------------------
    # Nodes are (frame_index, position, cloud_index); cloud_index is claimed
    # EAGERLY -- coarse endpoints at segment creation, intermediates at
    # attachment time -- so no two chains can ever race for one detection.
    # An unattached intermediate becomes a hard chain breaker.
    attached = missing = 0
    chains = []
    for tr in segments:
        times = [coarse_idx[int(t)] for t in tr["time"]]
        if len(times) < 2:
            continue

        def node(f: int, pos: np.ndarray, gate: float | None):
            """Claim the nearest free point within gate; None breaks chains."""
            nonlocal missing, attached
            k = claim(f, pos)
            if k >= 0 and (
                gate is None
                or float(np.linalg.norm(frame_particles[f][k] - pos)) <= gate
            ):
                used[f][k] = True
                if gate is not None:
                    attached += 1
                return (f, frame_particles[f][k], k)
            missing += 1
            return (f, None, -1)

        start = node(times[0], np.asarray(tr["pos"][0]), None)
        end = node(times[-1], np.asarray(tr["pos"][-1]), None)
        if start[2] < 0 or end[2] < 0:
            # Cannot anchor both endpoints; drop the segment rather than
            # fabricate a link whose endpoints are unknown detections.
            if start[2] >= 0:
                used[times[0]][start[2]] = False
            if end[2] >= 0:
                used[times[-1]][end[2]] = False
            missing += 1
            continue
        nodes: list[tuple[int, np.ndarray | None, int]] = [start]
        n_pairs = len(times) - 1
        for k in range(n_pairs):
            ta, tb = times[k], times[k + 1]
            pa, pb = np.asarray(tr["pos"][k]), np.asarray(tr["pos"][k + 1])
            va, vb = np.asarray(tr["vel"][k]), np.asarray(tr["vel"][k + 1])
            T = tb - ta
            d = pb - pa
            # Interior frames are at fractions k/T (the endpoints themselves
            # are detections already claimed at ta and tb).
            s = np.arange(1, T) / T
            h00 = 2 * s**3 - 3 * s**2 + 1
            h10 = s**3 - 2 * s**2 + s
            h01 = -2 * s**3 + 3 * s**2
            h11 = s**3 - s**2
            # A segment's very first point carries no velocity yet (the
            # coarse tracker seeds it as zero); a zero endpoint velocity
            # makes the Hermite sag below the true path, so substitute what
            # the segment itself implies.
            if not np.any(va):
                va = d / T
            if not np.any(vb):
                vb = d / T
            pred = (
                h00[:, None] * pa
                + h10[:, None] * (va * T)
                + h01[:, None] * pb
                + h11[:, None] * (vb * T)
            )
            for j, f in enumerate(range(ta + 1, tb)):
                nodes.append(node(f, pred[j], refine_gate))
            if k < n_pairs - 1:
                # Interior coarse point bridges this pair to the next one;
                # it belongs to this very track, so claim without a gate.
                nd = node(tb, pb, None)
                nodes.append(nd if nd[2] >= 0 else (tb, None, -1))
        nodes.append(end)

        # Emit maximal consecutive runs; every kept node carries its own
        # pre-claimed cloud index.
        run: list[tuple[int, np.ndarray, int]] = []

        def flush() -> None:
            if len(run) >= 2:
                chains.append(
                    {
                        "frame": [f for f, _, _ in run],
                        "idx": [k for _, _, k in run],
                        "pos": np.array([p for _, p, _ in run]),
                    }
                )

        for nd in nodes:
            if nd[1] is None:
                flush()
                run.clear()
                continue
            if run and nd[0] != run[-1][0] + 1:
                flush()
                run.clear()
            run.append(nd)
        flush()

    print(
        f"[hybrid_deltat_3d] coarse stride={stride}: {len(segments)} segment(s); "
        f"refine: {attached} intermediate point(s) attached, "
        f"{missing} left as gap(s); {len(chains)} chain(s) written"
    )
    return chains


class Tracking:
    """OpenPTV2 Tracking plugin interface for hybrid multi-Δt tracking."""

    def __init__(self, ptv=None, exp=None):
        self.ptv = ptv
        self.exp = exp

    def do_tracking(self) -> None:
        if self.exp is None:
            raise ValueError("No experiment object provided")

        print("Running Hybrid Multi-Delta-t 3D Tracking Plugin...")

        cpar = getattr(self.exp, "cpar", None)
        spar = getattr(self.exp, "spar", None)
        cals = getattr(self.exp, "cals", getattr(self.exp, "cal", []))
        res_dir = Path(getattr(self.exp, "res_dir", "res"))

        num_cams = cpar.num_cams if cpar is not None else (len(cals) if cals else 4)
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

        from openptv2.tracking_presets import unified_angle_deg, unified_velocity_bound

        dvxmax = unified_velocity_bound(track_cfg)
        dacc = float(track_cfg.get("dacc", 50.0))
        max_angle_deg = unified_angle_deg(track_cfg, default_deg=45.0)
        stride = int(track_cfg.get("stride", 5))
        refine_gate = float(track_cfg.get("refine_gate_mm", 0.8))

        from openptv2.gui.ptv import _open_run_store

        store = _open_run_store(self.exp)

        max_targets = 10000
        corres_base = str(res_dir / "rt_is")
        linkage_base = str(res_dir / "ptv_is")
        prio_base = str(res_dir / "added")

        frame_numbers = list(range(first_frame, last_frame + 1))

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
                store=store,
            )
            frames.append(frame)
            frame_particles.append(frame.positions())

        chains = hybrid_track(
            frame_particles,
            stride=stride,
            v_max=dvxmax,
            a_max=dacc,
            max_angle_deg=max_angle_deg,
            refine_gate=refine_gate,
        )

        total_links = 0
        for ch in chains:
            for k in range(len(ch["frame"]) - 1):
                li_cur, li_next = ch["frame"][k], ch["frame"][k + 1]
                frames[li_cur].path_next[ch["idx"][k]] = ch["idx"][k + 1]
                frames[li_next].path_prev[ch["idx"][k + 1]] = ch["idx"][k]
                total_links += 1

        for li, fn in enumerate(frame_numbers):
            frame = frames[li]
            frame._sync_soa_to_path()
            frame.write(
                corres_base,
                linkage_base,
                prio_file_base=prio_base,
                target_file_base=None,
                frame_num=fn,
                store=store,
            )

        num_frames = len(frame_numbers)
        total_particles = sum(f.num_parts for f in frames)
        n_steps = max(1, num_frames - 1)
        print(
            f"Average over sequence, particles: {total_particles / max(1, num_frames):.1f}, "
            f"links: {total_links / n_steps:.1f}, "
            f"lost: {total_particles / max(1, num_frames) - total_links / n_steps:.1f}"
        )
        print(
            "[hybrid_deltat_3d] note: chains break where an intermediate frame "
            "had no detection within refine_gate_mm of the prediction; "
            "postptv stitching (max_gap>=1) can re-glue those gaps."
        )
        print("Hybrid Multi-Delta-t 3D Tracking completed successfully.")
