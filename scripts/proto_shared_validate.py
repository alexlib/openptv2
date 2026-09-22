"""Shared-observation prototype validation: one 3D scene, three trackers.

Scene (10 frames, mm units, constant velocity + jitter):
  P0/P1: crossing pair, frames 4-5 occluded -> single midpoint point.
  P2:    dropout gap at frames 3-4 (no point at all).
  P3:    exits the volume after frame 6 (legal end).
  P4/P5: clean crucirng traffic.
Ground truth = 6 trajectories. Metrics: tracks, switches, missing, purity.
"""
import sys

import numpy as np

N_FRAMES = 10


def make_scene(seed=0, jitter=0.05):
    rng = np.random.default_rng(seed)
    vel = {
        0: np.array([2.0, 0.4, 0.0]),
        1: np.array([2.0, -0.4, 0.0]),
        2: np.array([1.0, 1.5, 0.2]),
        3: np.array([3.0, 0.0, 0.0]),
        4: np.array([-1.5, 1.0, 0.1]),
        5: np.array([0.5, -1.2, -0.1]),
    }
    p0 = {
        0: np.array([0.0, -2.0, 0.0]),
        1: np.array([0.0, 2.0, 0.0]),
        2: np.array([5.0, 5.0, 1.0]),
        3: np.array([0.0, 10.0, -1.0]),
        4: np.array([20.0, 0.0, 2.0]),
        5: np.array([8.0, 12.0, 0.5]),
    }
    # P0/P1 cross near x=10 around frame 5: shift so midpoint lands ~frame 4-5
    frames_pts, frames_ids = [], []  # per frame: (M,3) points, (M,) truth ids
    for f in range(N_FRAMES):
        pts, ids = [], []
        for i in range(6):
            if i == 3 and f > 6:
                continue  # legal exit
            if i == 2 and f in (3, 4):
                continue  # dropout gap
            pos = p0[i] + vel[i] * f + rng.normal(0, jitter, 3)
            pts.append(pos)
            ids.append(i)
        pts = np.array(pts)
        ids = np.array(ids)
        if f in (4, 5):
            # occlusion: P0 and P1 merge into their midpoint (single point)
            m0 = np.where(ids == 0)[0]
            m1 = np.where(ids == 1)[0]
            if len(m0) and len(m1):
                mid = 0.5 * (pts[m0[0]] + pts[m1[0]])
                keep = [k for k in range(len(ids)) if k not in (m0[0], m1[0])]
                pts = np.vstack([pts[keep], mid[None, :]])
                ids = np.append(ids[keep], -1)  # -1 = merged, both truths
        frames_pts.append(pts)
        frames_ids.append(ids)
    return frames_pts, frames_ids


def score_tracks(tracks, frames_ids, frames_pts, tol=1.0):
    """tracks: list of dicts with 'frames' (time idx) and 'pos' (xyz).

    Coverage counts UNIQUE truth points (two tracks may honestly cover one
    shared detection). Returns (n_tracks, switches, missing, pure)."""
    truth_total = sum(int((ids >= 0).sum()) for ids in frames_ids)
    # +2 merged points per occlusion frame counted separately
    truth_total += sum(int((ids == -1).sum()) * 2 for ids in frames_ids)
    switches, pure = 0, 0
    covered: dict[tuple[int, int], set[int]] = {}  # (frame,row) -> track idx
    for ti, t in enumerate(tracks):
        seen: set[int] = set()
        prev = None
        for f, p in zip(t["frames"], t["pos"]):
            ids = frames_ids[f]
            pts = frames_pts[f]
            if len(ids) == 0:
                continue
            d = np.linalg.norm(pts - p, axis=1)
            b = int(np.argmin(d))
            if d[b] > tol:
                continue
            covered.setdefault((int(f), int(b)), set()).add(ti)
            if ids[b] == -1:
                seen |= {0, 1}
                truth = {0, 1}
            else:
                seen.add(int(ids[b]))
                truth = {int(ids[b])}
            if prev is not None and not truth & prev:
                switches += 1
            prev = truth if prev is None else (prev | truth)
        if seen <= {0, 1} or len(seen) == 1:
            pure += 1
    # unique truth points covered: a merged row counts double only when two
    # tracks honestly hold it (shared observation), else single.
    uniq = 0
    for (f, r), holders in covered.items():
        if r == -1 and len(holders) >= 2:
            uniq += 2
        else:
            uniq += 1
    return {
        "n_tracks": len(tracks),
        "switches": switches,
        "missing": max(0, truth_total - uniq),
        "pure": pure,
    }


def to_links_format(tracks, key_frames="frames", key_pos="pos"):
    return tracks


def fast3d_tracks(frame_particles, v_max=3.0, share_tol=0.0):
    """Drive track3d_loop_fast directly (mirrors Cython3DTracker.track_frames)
    with extended per-frame arrays. Shared pairs materialize as virtual
    carrier particles: shared coordinates, own history (prev -> frame-t
    detection). Returns (tracks, n_shared)."""
    from openptv2.algorithms.constants import NEXT_NONE, PREV_NONE
    from openptv2.algorithms.track_kernels_track3d import track3d_loop_fast

    nf = len(frame_particles)
    ext_pos = [np.ascontiguousarray(p, dtype=np.float64)
               for p in frame_particles]
    ext_prev = [np.full(len(p), PREV_NONE, dtype=np.int32)
                for p in frame_particles]
    ext_next = [np.full(len(p), NEXT_NONE, dtype=np.int32)
                for p in frame_particles]
    n_shared_total = 0
    for t in range(nf - 1):
        if t == 0:
            n0 = 0
            pos_0 = np.empty((0, 3), dtype=np.float64)
            prev_0 = np.empty(0, dtype=np.int32)
        else:
            n0 = len(ext_pos[t - 1])
            pos_0, prev_0 = ext_pos[t - 1], ext_prev[t - 1]
        n1, n2 = len(ext_pos[t]), len(ext_pos[t + 1])
        if n1 == 0 or n2 == 0:
            continue
        pos_1, prev_1 = ext_pos[t], ext_prev[t]
        nxt_1 = ext_next[t]
        pos_2 = ext_pos[t + 1]
        prev_2 = ext_prev[t + 1]
        nxt_2 = ext_next[t + 1]
        cap = max(n2, 1)
        sc = np.zeros(1, dtype=np.int32)
        si = np.full(cap, -1, dtype=np.int32)
        sk = np.full(cap, -1, dtype=np.int32)
        track3d_loop_fast(
            n1, pos_0, prev_0, n0, pos_1, prev_1, nxt_1, n1,
            pos_2, prev_2, nxt_2, n2,
            v_max, v_max, v_max, 32, 0.0,
            share_tol, sc, si, sk,
        )
        ext_next[t] = nxt_1
        ext_prev[t + 1] = prev_2
        # materialize virtual carriers for the NEXT steps (curr/prev roles)
        for e in range(int(sc[0])):
            i, k = int(si[e]), int(sk[e])
            ext_pos[t + 1] = np.vstack(
                [ext_pos[t + 1], pos_2[k][None, :]])
            ext_prev[t + 1] = np.append(ext_prev[t + 1], np.int32(i))
            ext_next[t + 1] = np.append(ext_next[t + 1], np.int32(NEXT_NONE))
            n_shared_total += 1
    # assemble (shared carriers emit their coords into their track)
    counts = [len(p) for p in ext_pos]
    visited = [np.zeros(n, dtype=bool) for n in counts]
    tracks = []
    for t in range(nf):
        for i in range(counts[t]):
            if visited[t][i]:
                continue
            tr_pos, tr_time, ct, ci = [], [], t, i
            while ct < nf and ci != NEXT_NONE and not visited[ct][ci]:
                visited[ct][ci] = True
                tr_pos.append(ext_pos[ct][ci])
                tr_time.append(ct)
                nx = int(ext_next[ct][ci])
                if nx >= 0 and ct + 1 < nf:
                    ct, ci = ct + 1, nx
                else:
                    break
            if len(tr_pos) >= 2:
                tracks.append({"frames": tr_time,
                               "pos": np.array(tr_pos)})
    return tracks, n_shared_total


def trackcorr_linkage_tracks(frame_particles, base_links, tol=1.0):
    """Prototype trackcorr path: base_links (greedy forward output) -> plain
    linkage arrays -> mark_shared_observations -> assemble_with_shared.
    Returns (tracks_plain, tracks_shared, n_marks)."""
    from openptv2.tracking_shared import (
        mark_shared_observations,
        assemble_with_shared,
    )
    from openptv2.algorithms.constants import PREV_NONE, NEXT_NONE

    nf = len(frame_particles)
    frames = {}
    for f in range(nf):
        n = len(frame_particles[f])
        frames[f] = (np.full(n, PREV_NONE, dtype=np.int32),
                     np.full(n, NEXT_NONE, dtype=np.int32),
                     np.asarray(frame_particles[f], dtype=np.float64))
    # row bookkeeping: links reference (frame, row); rows == indices here
    for t0, r0, t1, r1 in base_links:
        _, nxt, _ = frames[t0]
        prv, _, _ = frames[t1]
        if r0 < len(nxt) and r1 < len(prv):
            nxt[r0] = r1
            prv[r1] = r0
    plain = assemble_with_shared(frames, 0, nf - 1, {})
    shared = mark_shared_observations(frames, 0, nf - 1, tol=tol)
    healed = assemble_with_shared(frames, 0, nf - 1, shared)
    return plain, healed, len(shared)


def greedy_links(frame_particles, gate=4.0):
    """Baseline forward pass: nearest-neighbour chains (what trackcorr's
    kernel would produce before postprocess). Returns link tuples."""
    links = []
    for t in range(len(frame_particles) - 1):
        p0 = np.asarray(frame_particles[t])
        p1 = np.asarray(frame_particles[t + 1])
        used = set()
        for i in range(len(p0)):
            best, bj = gate, -1
            for j in range(len(p1)):
                if j in used:
                    continue
                d = float(np.linalg.norm(p1[j] - p0[i]))
                if d < best:
                    best, bj = d, j
            if bj >= 0:
                used.add(bj)
                links.append((t, i, t + 1, bj))
    return links


if __name__ == "__main__":
    from openptv2.plugins.two_phase_tracking import (
        TwoPhaseTracker,
        TwoPhaseTrackerConfig,
    )

    fp, fi = make_scene()
    print("dets/frame:", [len(p) for p in fp], "truth(server): 6 trajs")
    for share in [False, True]:
        cfg = TwoPhaseTrackerConfig(v_max=4.0, max_gap=2, dt=1.0,
                                    leaf_weight=0.0, cost_mode="3d",
                                    use_velocity=True, allow_shared=share,
                                    max_shared=3)
        tr = TwoPhaseTracker(cfg)
        links, chains = tr.track_frames([np.asarray(p) for p in fp],
                                        return_chains=True)
        tracks = [{"frames": c["frames"], "pos": c["pos"]} for c in chains
                  if len(c["frames"]) >= 1]
        n_shared = sum(sum(c["shared"]) for c in chains)
        print(f"allow_shared={share}: links={len(links)} shared_pts={n_shared}",
              score_tracks(tracks, fi, fp))

    print("--- Fast3D (track3d_loop_fast, virtual carriers) ---")
    for tol in [0.0, 0.3, 0.5, 1.0]:
        tr, ns = fast3d_tracks([np.asarray(p) for p in fp], v_max=3.0,
                               share_tol=tol)
        print(f"share_tol={tol}: shared={ns}",
              score_tracks(tr, fi, fp))

    print("--- trackcorr-linkage (greedy links -> mark+assemble) ---")
    base = greedy_links([np.asarray(p) for p in fp])
    plain, healed, nm = trackcorr_linkage_tracks(
        [np.asarray(p) for p in fp], base, tol=1.0)
    print(f"plain: {score_tracks(plain, fi, fp)}")
    print(f"shared marks={nm}: {score_tracks(healed, fi, fp)}")
