"""How often can a merged blob observation be resolved from other cameras / other frames?
(docs/plans/2026-11-19-joint-tracker.md, section 15). Truth-based, on the benchmark cases:

    uv run python scripts/joint_info_coverage.py L1_d1_c0_k1_n200_s0.08_a0.18_f0.05

For every merged observation (particle, frame, camera: its blob is shared with another particle)
it reports whether the same particle has >= 2 clean cameras in the same frame, a clean blob in the same
camera within +-T frames, or >= 2 clean cameras (a 3D position) at a nearby frame.
"""
import sys, numpy as np
from pathlib import Path
W = Path.home() / "Downloads/HiDImaging/tracker-bench-2026-09-29/cases"
def run(case):
    tr = np.load(W / case / "truth.npz")
    fr, pid, blob = tr["frame"], tr["particle_id"], tr["blob_tid"]
    n = len(fr); C = blob.shape[1]
    det = blob >= 0
    shared = np.zeros_like(det)
    for f in np.unique(fr):
        idx = np.flatnonzero(fr == f)
        for c in range(C):
            _u, inv, cnt = np.unique(blob[idx, c], return_inverse=True, return_counts=True)
            shared[idx, c] = det[idx, c] & (cnt[inv] > 1)
    clean = det & ~shared
    key = pid.astype(np.int64) * 100000 + fr
    order = np.argsort(key); skey = key[order]
    def row_of(p, f):
        k = p.astype(np.int64) * 100000 + f
        pos = np.searchsorted(skey, k); pos = np.minimum(pos, n - 1)
        ok = skey[pos] == k
        return np.where(ok, order[pos], -1)
    res = {}
    merged = shared                                    # (n, C) merged observations
    r_idx, c_idx = np.nonzero(merged)
    tot = len(r_idx)
    # cross-camera: other cameras of the same particle and frame with a clean blob
    oth = clean[r_idx].copy(); oth[np.arange(tot), c_idx] = False
    cross2 = oth.sum(1) >= 2
    # temporal: same camera, the same particle clean within +-T frames
    out = {"merged_obs": tot, "share_of_detected_obs": tot / det.sum(), "cross_clean>=2": float(cross2.mean())}
    for T in (1, 3, 6):
        tmp = np.zeros(tot, bool)
        for d in list(range(-T, 0)) + list(range(1, T + 1)):
            rr = row_of(pid[r_idx], fr[r_idx] + d)
            ok = rr >= 0
            tmp[ok] |= clean[rr[ok], c_idx[ok]]
        out[f"temporal_clean_within_{T}"] = float(tmp.mean())
        out[f"either_T{T}"] = float((tmp | cross2).mean())
        out[f"neither_T{T}"] = float((~(tmp | cross2)).mean())
        if T == 3: tmp3 = tmp
    # 3D view: the particle has >= 2 clean cameras (so a 3D position) at a nearby frame, any camera;
    # "both sides": such a frame before AND after (interpolation instead of extrapolation)
    ok3 = clean.sum(1) >= 2
    for T in (1, 3, 6):
        before = np.zeros(tot, bool); after = np.zeros(tot, bool); same = cross2.copy()
        for d in range(1, T + 1):
            rb = row_of(pid[r_idx], fr[r_idx] - d); ra = row_of(pid[r_idx], fr[r_idx] + d)
            before |= (rb >= 0) & ok3[np.maximum(rb, 0)] & (rb >= 0)
            after |= (ra >= 0) & ok3[np.maximum(ra, 0)] & (ra >= 0)
        out[f"3d_known_same_or_near_T{T}"] = float((same | before | after).mean())
        out[f"3d_known_both_sides_T{T}"] = float((before & after).mean())
        out[f"3d_unknown_T{T}"] = float((~(same | before | after)).mean())
    # particle-level: merged in a camera but clean in >=2 other cameras in the same frame => the 3D point is
    # well defined; share of 3-camera-or-more particles with at least one merged camera that still keep >=3 clean cams
    nclean = clean.sum(1); ndet = det.sum(1); nmerged = shared.sum(1)
    sel = ndet >= 3
    out["particles_ge3cams"] = int(sel.sum())
    out["share_with_a_merged_cam"] = float((nmerged[sel] > 0).mean())
    out["share_with_>=3_clean_cams"] = float((nclean[sel] >= 3).mean())
    out["share_with_>=2_clean_cams"] = float((nclean[sel] >= 2).mean())
    out["share_with_<2_clean_cams"] = float((nclean[sel] < 2).mean())
    return out
for c in sys.argv[1:]:
    print(c); [print("  %-34s %s" % (k, ("%.3f" % v) if isinstance(v, float) else v)) for k, v in run(c).items()]
