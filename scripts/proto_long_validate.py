"""Long-window validation (50 real frames) for the three shared prototypes.

Data: HiDImaging wp1 test, frames 100001-100050, ~1000 pts/frame.
Input: res_orig/rt_is.* 3D points. Reference: res_orig/ptv_is.* linkages
(0-based prev/next chains) = the res_orig tracking output itself.
Prototypes (worktree code): TwoPhase +/-share, Fast3D +/-share_tol,
greedy baseline vs +mark/assemble (trackcorr-linkage path).
Metrics: tracks, mean length, coverage, fragmentation of reference tracks,
impurity of prototype tracks (switch/merge proxy), shared events, runtime.
"""
import sys
import time

import numpy as np

TEST = r"C:\Users\alex\Downloads\HiDImaging\CompleteTest\wp1\test"
F0, NF = 100001, 50
TOL = 1.0  # mm


def load_points():
    frames = []
    for f in range(F0, F0 + NF):
        rows = []
        with open(f"{TEST}/res_orig/rt_is.{f}") as fh:
            n = int(fh.readline().split()[0])
            for _ in range(n):
                t = fh.readline().split()
                rows.append([float(t[1]), float(t[2]), float(t[3])])
        frames.append(np.array(rows))
    return frames


def load_linkages():
    prev, nxt, pos = {}, {}, {}
    for f in range(F0, F0 + NF):
        with open(f"{TEST}/res_orig/ptv_is.{f}") as fh:
            n = int(fh.readline().split()[0])
            pv = np.full(n, -1, np.int32)
            nx = np.full(n, -2, np.int32)
            xy = np.zeros((n, 3))
            for i in range(n):
                t = fh.readline().split()
                pv[i], nx[i] = int(t[0]), int(t[1])
                xy[i] = [float(t[2]), float(t[3]), float(t[4])]
        prev[f], nxt[f], pos[f] = pv, nx, xy
    return prev, nxt, pos


def assemble_ref(prev, nxt, pos):
    chains = []
    visited = set()
    for f in range(F0, F0 + NF):
        for i in range(len(pos[f])):
            if (f, i) in visited or int(prev[f][i]) >= 0:
                continue
            chain = []
            cf, ci = f, i
            while True:
                if (cf, ci) in visited:
                    break
                visited.add((cf, ci))
                chain.append((cf, ci))
                if cf not in nxt or ci >= len(nxt[cf]):
                    break
                ni = int(nxt[cf][ci])
                if ni < 0 or cf + 1 not in pos or ni >= len(pos[cf + 1]):
                    break
                cf += 1
                ci = ni
            if len(chain) >= 1:
                chains.append(chain)
    # orphans (all-prev/no-next singletons not yet visited)
    for f in range(F0, F0 + NF):
        for i in range(len(pos[f])):
            if (f, i) not in visited:
                chains.append([(f, i)])
                visited.add((f, i))
    return chains, pos


def calibrate(chains, pos):
    steps = []
    for c in chains:
        for a in range(1, len(c)):
            f0, i0 = c[a - 1]
            f1, i1 = c[a]
            if f1 == f0 + 1:
                steps.append(float(np.linalg.norm(pos[f1][i1] - pos[f0][i0])))
    steps = np.array(steps)
    vm = float(np.percentile(steps, 99))
    print(f"calibrate: {len(chains)} ref chains, step p50="
          f"{np.median(steps):.3f} p99={vm:.3f}", flush=True)
    return vm


def score(proto_tracks, ref_chains, pos):
    from scipy.spatial import cKDTree
    # frame -> (points, proto_tid) / (points, ref_chain_idx) indexes
    ptrees, rtrees = {}, {}
    pmap, rmap = {}, {}
    for ti, t in enumerate(proto_tracks):
        for f, p in zip(t["frames"], t["pos"]):
            pmap.setdefault(int(f), []).append((np.asarray(p), ti))
    for ci, c in enumerate(ref_chains):
        for (f, i) in c:
            rmap.setdefault(f, []).append((pos[f][i], ci))
    for f in set(list(pmap) + list(rmap)):
        if f in pmap:
            pts = np.array([p for p, _ in pmap[f]])
            ptrees[f] = (cKDTree(pts), [ti for _, ti in pmap[f]])
        if f in rmap:
            pts = np.array([p for p, _ in rmap[f]])
            rtrees[f] = (cKDTree(pts), [ci for _, ci in rmap[f]])
    # fragmentation: ref chains split across >1 proto tracks
    frag, frag_den, cover_pts, cover_den = 0, 0, 0, 0
    for c in [c for c in ref_chains if len(c) >= 3]:
        owners = set()
        for (f, i) in c:
            cover_den += 1
            if f not in ptrees:
                continue
            tree, tids = ptrees[f]
            d, k = tree.query(pos[f][i], k=1, distance_upper_bound=TOL)
            if d <= TOL:
                owners.add(tids[int(k)])
                cover_pts += 1
        frag_den += 1
        if len(owners) > 1:
            frag += 1
    # impurity: proto tracks spanning >1 ref chain
    impure = 0
    for t in proto_tracks:
        if len(t["frames"]) < 2:
            continue
        owners = set()
        for f, p in zip(t["frames"], t["pos"]):
            if int(f) not in rtrees:
                continue
            tree, cis = rtrees[int(f)]
            d, k = tree.query(np.asarray(p), k=1, distance_upper_bound=TOL)
            if d <= TOL:
                owners.add(cis[int(k)])
        if len(owners) > 1:
            impure += 1
    ntrk = sum(1 for t in proto_tracks if len(t["frames"]) >= 2)
    ml = np.mean([len(t["frames"]) for t in proto_tracks
                  if len(t["frames"]) >= 2]) if ntrk else 0.0
    return {"ntrk": ntrk, "meanlen": round(float(ml), 2),
            "cover": round(cover_pts / max(cover_den, 1), 3),
            "frag%": round(100 * frag / max(frag_den, 1), 1),
            "impure": impure}


def main():
    t0 = time.perf_counter()
    frames = load_points()
    prev, nxt, pos = load_linkages()
    ref_chains, _ = assemble_ref(prev, nxt, pos)
    v_max = calibrate(ref_chains, pos)
    print(f"load done ({time.perf_counter() - t0:.1f}s)", flush=True)
    ref_tracks = [{"frames": [f for f, _ in c],
                   "pos": np.array([pos[f][i] for f, i in c])}
                  for c in ref_chains]
    print(f"res_orig: {score(ref_tracks, ref_chains, pos)}", flush=True)

    from openptv2.plugins.two_phase_tracking import (
        TwoPhaseTracker,
        TwoPhaseTrackerConfig,
    )

    for share in [False, True]:
        t1 = time.perf_counter()
        cfg = TwoPhaseTrackerConfig(v_max=v_max, max_gap=2, dt=1.0,
                                    leaf_weight=0.0, cost_mode="3d",
                                    use_velocity=True, allow_shared=share)
        _, chains = TwoPhaseTracker(cfg).track_frames(
            [np.asarray(p) for p in frames], return_chains=True)
        tracks = [{"frames": [F0 + f for f in c["frames"]], "pos": c["pos"]}
                  for c in chains if len(c["frames"]) >= 1]
        ns = sum(sum(c["shared"]) for c in chains)
        print(f"twophase share={share}: shared_pts={ns} "
              f"{score(tracks, ref_chains, pos)} "
              f"({time.perf_counter() - t1:.1f}s)", flush=True)

    sys.path.insert(0, "scripts")
    from proto_shared_validate import fast3d_tracks, trackcorr_linkage_tracks
    from scipy.spatial import cKDTree

    # Fast3D pure-Python kernel: 15 frames (50 would take too long in
    # interpreted mode; the compiled path is unaffected).
    NF_FAST = 15
    pos15 = {f: pos[f] for f in range(F0, F0 + NF_FAST)}
    chains15 = []
    for c in ref_chains:
        cc = [(f, i) for (f, i) in c if F0 <= f < F0 + NF_FAST]
        if cc:
            chains15.append(cc)
    for tol in [0.0, 0.5]:
        t1 = time.perf_counter()
        tr, ns = fast3d_tracks([np.asarray(p) for p in frames[:NF_FAST]],
                               v_max=v_max, share_tol=tol)
        for t in tr:
            t["frames"] = [F0 + f for f in t["frames"]]
        print(f"fast3d[{NF_FAST}f] tol={tol}: shared={ns} "
              f"{score(tr, chains15, pos15)} "
              f"({time.perf_counter() - t1:.1f}s)", flush=True)

    def greedy_links_kd(frame_particles, gate):
        links = []
        for t in range(len(frame_particles) - 1):
            p0 = np.asarray(frame_particles[t])
            p1 = np.asarray(frame_particles[t + 1])
            if len(p0) == 0 or len(p1) == 0:
                continue
            tree = cKDTree(p1)
            dists, idxs = tree.query(p0, k=1, distance_upper_bound=gate)
            used = set()
            for i in range(len(p0)):
                j = int(idxs[i])
                if j < len(p1) and j not in used and dists[i] <= gate:
                    used.add(j)
                    links.append((t, i, t + 1, j))
        return links

    t1 = time.perf_counter()
    base = greedy_links_kd([np.asarray(p) for p in frames], gate=v_max)
    plain, healed, nm = trackcorr_linkage_tracks(
        [np.asarray(p) for p in frames], base, tol=1.0)
    for name, tr in [("plain", plain), ("healed", healed)]:
        for t in tr:
            t["frames"] = [F0 + f for f in t["frames"]]
        print(f"tclink {name}: marks={nm if name == 'healed' else 0} "
              f"{score(tr, ref_chains, pos)} "
              f"({time.perf_counter() - t1:.1f}s)", flush=True)

    print("done.")


if __name__ == "__main__":
    main()
