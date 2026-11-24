# ruff: noqa
"""Where does the tracker lose particles on syn1, and how much of it sits inside a merge episode a streak could bridge?

    uv run python bench/limits/loss_in_episodes.py <run dir with res/run.zarr> <n_frames> [K]

Truth side (per particle, frame, camera): the blob holding the particle (nearest within 3 px) and how many particles it holds
(m): 'clean' m == 1, 'shared' m >= 2, 'noblob' no blob. A shared view is 'bridgeable' when the particle has a clean view in
the same camera within K frames on both sides (interpolation) or on one side (extrapolation).
Tracker side: the points of the run's linkage matched to truth like synth_bench.evaluate_pred (bias removed, eps 0.4 mm).
Classes of a particle-frame that the tracker lost (or holds with fewer than 4 cameras):
  R0 clean in >=3 cameras      -> not a merge problem (matcher / linker)
  R1 needs shared views, all bridgeable on both sides      -> a streak could supply them
  R2 ... with at least one-sided bridges only
  R3 >=3 blobs but a shared view has no clean frame within K -> episode too long
  R4 fewer than 3 blobs (faint / outside the view)
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import synth_bench as sb
from openptv2.merge_post import arrays_from_store
from openptv2.storage import RunStore

run, N = Path(sys.argv[1]), int(sys.argv[2])
K = int(sys.argv[3]) if len(sys.argv) > 3 else 6
st = RunStore(str(run / "res" / "run.zarr"), mode="r")
a = arrays_from_store(st, "ptv_is", 1, N, 4)
tr = np.load(sb.SRC / "truth" / "truth.npz")
keep = tr["frame"] <= N
t_id, t_fr, t_pos, t_px = tr["particle_id"][keep], tr["frame"][keep], tr["pos_mm"][keep], tr["pixel_xy_per_cam"][keep]
by_frame = {int(f): (t_pos[t_fr == f], t_id[t_fr == f], cKDTree(t_pos[t_fr == f])) for f in np.unique(t_fr)}

# ---- tracker points matched to truth (as evaluate_pred)
g0, _, off = sb._match(by_frame, a["frame"], a["pos"], 0.6)
bias = np.median(off[g0 >= 0], axis=0)
gid, _, _ = sb._match(by_frame, a["frame"], a["pos"] - bias, 0.4)
ncam_pt = (a["blob"] >= 0).sum(1)
used = a["blob"]                                  # blob id per camera of every output point
found = {}                                        # (pid, frame) -> output row
for i in np.flatnonzero(gid >= 0):
    found[int(gid[i]), int(a["frame"][i])] = i
tid_of = {k: int(a["tid"][i]) for k, i in found.items()} if "tid" in a else {}
if not tid_of:
    raise SystemExit("no trajectory ids in arrays_from_store")

# ---- truth blob membership
nb = {}                                           # (frame, cam) -> blob table
bidx = np.full((len(t_id), 4), -1); mcnt = np.zeros((len(t_id), 4), int)
for f in np.unique(t_fr):
    rows_f = np.flatnonzero(t_fr == f)
    for c in range(4):
        r = np.asarray(st.root[f"targets/cam_{c}/frame_{int(f):06d}"])
        px = t_px[rows_f][:, 2 * c: 2 * c + 2]
        ok = np.isfinite(px[:, 0])
        d, j = cKDTree(r[:, 1:3]).query(px[ok], distance_upper_bound=3.0)
        has = np.isfinite(d)
        sel = rows_f[ok][has]
        bidx[sel, c] = j[has]
        cnt = np.bincount(j[has], minlength=len(r))
        mcnt[sel, c] = cnt[j[has]]
inview = np.isfinite(t_px[:, [0, 2, 4, 6]])
clean = (bidx >= 0) & (mcnt == 1); shared = (bidx >= 0) & (mcnt >= 2); noblob = inview & (bidx < 0)

# clean frames per (particle, camera)
cf = defaultdict(list)
for i in np.flatnonzero(clean.any(1)):
    for c in np.flatnonzero(clean[i]):
        cf[int(t_id[i]), int(c)].append(int(t_fr[i]))
cf = {k: np.array(sorted(v)) for k, v in cf.items()}

def bridge(p, c, f):
    v = cf.get((p, c))
    if v is None: return 0, 0
    k = np.searchsorted(v, f)
    left = k > 0 and f - v[k - 1] <= K
    right = k < len(v) and v[k] - f <= K        # v[k] > f since the particle is shared at f
    return int(left and right), int(left or right)

rows = np.flatnonzero((t_fr > K) & (t_fr <= N - K) & (inview.sum(1) >= 3))
klass = {}; state = {}
for i in rows:
    p, f = int(t_id[i]), int(t_fr[i])
    state[p, f] = int(ncam_pt[found[p, f]]) if (p, f) in found else 0
    ncl = int(clean[i].sum())
    sh = np.flatnonzero(shared[i])
    both = sum(bridge(p, int(c), f)[0] for c in sh); one = sum(bridge(p, int(c), f)[1] for c in sh)
    if ncl >= 3: k = "R0 clean in >=3 cameras (not a merge problem)"
    elif ncl + both >= 3: k = "R1 a streak could supply it (clean both sides within K)"
    elif ncl + one >= 3: k = "R2 one-sided streak only"
    elif ncl + len(sh) >= 3: k = "R3 shared, no clean frame within K (episode too long)"
    else: k = "R4 fewer than 3 blobs (faint / outside)"
    klass[p, f] = k

n = len(rows)
print(f"[{run.name}] truth particle-frames (>=3 cameras in view, frames {K+1}-{N-K}): {n}")
st_arr = np.array([state[int(t_id[i]), int(t_fr[i])] for i in rows])
print("  tracker state: lost %.1f%%   3 cameras %.1f%%   4 cameras %.1f%%   (2 cameras %.1f%%)" % tuple(100 * np.array([(st_arr == 0).mean(), (st_arr == 3).mean(), (st_arr == 4).mean(), (st_arr == 2).mean()])))
ck = [klass[int(t_id[i]), int(t_fr[i])] for i in rows]
for title, mask in (("LOST particle-frames", st_arr == 0), ("held with 3 cameras (a view missing)", st_arr == 3)):
    c = Counter(np.array(ck)[mask]); tot = mask.sum()
    print(f"  {title}: {tot} ({100 * tot / n:.1f}% of all)")
    for kk in sorted(c): print("     %-62s %6d  %5.1f%%" % (kk, c[kk], 100 * c[kk] / tot))

# ---- track breaks: per true particle, gaps between found frames and id switches
gaps = []; switches = 0; pieces = 0; npart = 0
byp = defaultdict(list)
for (p, f) in klass: byp[p].append(f)
for p, fs in byp.items():
    fs = sorted(fs)
    if len(fs) < 2 * K: continue
    npart += 1
    prev = None
    for f in fs:
        if (p, f) in found:
            if prev is not None:
                pf, pt = prev
                if f - pf > 1:
                    gaps.append((p, pf, f))
                elif tid_of[p, f] != pt:
                    switches += 1
            prev = (f, tid_of[p, f])
print(f"  particles followed {npart}; breaks inside found stretches: {len(gaps)} gaps (lost frames between two found frames), {switches} id switches with every frame found")
gc = Counter()
for p, f0, f1 in gaps:
    ks = [klass.get((p, f), "?") for f in range(f0 + 1, f1) if (p, f) in klass]
    if not ks: continue
    if all(k.startswith(("R1",)) for k in ks): g = "gap fully in R1 (streak could bridge)"
    elif all(k.startswith(("R1", "R2")) for k in ks): g = "gap in R1/R2 (streak, maybe one-sided)"
    elif any(k.startswith("R0") for k in ks) and not any(k.startswith(("R3", "R4")) for k in ks): g = "gap with clean-in-3-cameras frames (matcher / linker)"
    elif any(k.startswith("R4") for k in ks): g = "gap with a frame of fewer than 3 blobs (faint)"
    else: g = "gap with an over-long episode (R3)"
    gc[g] += 1
for kk in sorted(gc): print("     %-62s %6d  %5.1f%%" % (kk, gc[kk], 100 * gc[kk] / max(len(gaps), 1)))

# ---- id switches with every frame found: is a merge involved?
row_of = {(int(t_id[i]), int(t_fr[i])): i for i in np.flatnonzero(inview.sum(1) >= 3)}
sw = Counter()
for p, fs in byp.items():
    fs = sorted(fs)
    for f0, f1 in zip(fs[:-1], fs[1:]):
        if f1 - f0 != 1 or (p, f0) not in found or (p, f1) not in found: continue
        if tid_of[p, f0] == tid_of[p, f1]: continue
        i0, i1 = row_of[p, f0], row_of[p, f1]
        sh = shared[i0].any() or shared[i1].any()
        n0, n1 = ncam_pt[found[p, f0]], ncam_pt[found[p, f1]]
        # wrong link (tracker joined to another particle)? the new id's neighbour: does the other end belong to the same truth id?
        if sh: kk = "a camera shares the blob at one of the two frames"
        elif min(n0, n1) < 4: kk = "no shared blob, but a point has only 3 cameras"
        else: kk = "clean 4-camera points on both frames (linker only)"
        sw[kk] += 1
tot = sum(sw.values())
print(f"  id switches with both frames found, one frame apart: {tot}")
for kk in sorted(sw): print("     %-62s %6d  %5.1f%%" % (kk, sw[kk], 100 * sw[kk] / tot))

# ---- cause of the missing view of 3-camera points
vc = Counter()
for i in rows:
    p, f = int(t_id[i]), int(t_fr[i])
    if state[p, f] != 3: continue
    r = found[p, f]
    for c in range(4):
        if used[r, c] >= 0 or not inview[i, c]: continue
        if noblob[i, c]: vc["no blob there (faint)"] += 1
        elif shared[i, c]:
            b, o = bridge(p, c, f)
            vc["shared blob, clean frame both sides within K" if b else ("shared blob, one side only" if o else "shared blob, no clean frame within K")] += 1
        else: vc["clean blob exists but was not used"] += 1
tot = sum(vc.values())
print(f"  missing view of a 3-camera point, by cause: {tot}")
for kk in sorted(vc): print("     %-62s %6d  %5.1f%%" % (kk, vc[kk], 100 * vc[kk] / tot))

# ---- base rates: the same quantities for pairs that did NOT break, and for all rows
ok_sh = ok_n = 0
for p, fs in byp.items():
    fs = sorted(fs)
    for f0, f1 in zip(fs[:-1], fs[1:]):
        if f1 - f0 != 1 or (p, f0) not in found or (p, f1) not in found: continue
        if tid_of[p, f0] != tid_of[p, f1]: continue
        i0, i1 = row_of[p, f0], row_of[p, f1]
        ok_n += 1; ok_sh += bool(shared[i0].any() or shared[i1].any())
print(f"  base rate: pairs linked correctly with a shared blob in some camera: {100 * ok_sh / ok_n:.1f}% of {ok_n}   (switches: {100 * sw['a camera shares the blob at one of the two frames'] / tot_sw:.1f}%)" if (tot_sw := sum(sw.values())) else "")
allsh = shared[rows].any(1)
print(f"  base rate: all particle-frames with a shared blob in some camera {100 * allsh.mean():.1f}%; lost rows {100 * allsh[st_arr == 0].mean():.1f}%; held with 4 cameras {100 * allsh[st_arr == 4].mean():.1f}%; held with 3 cameras {100 * allsh[st_arr == 3].mean():.1f}%")
nsh = shared[rows].sum(1)
for k in range(5): print(f"     rows with {k} shared cameras: {100 * (nsh == k).mean():5.1f}% of all, tracker lost {100 * (st_arr[nsh == k] == 0).mean():5.1f}% of them, held with 4 cameras {100 * (st_arr[nsh == k] == 4).mean():5.1f}%")

# ---- gap length: the scored pipeline already repairs gaps of up to 3 frames in 3D (repair_arrays, max_gap=3)
gl = np.array([f1 - f0 - 1 for p, f0, f1 in gaps])
print(f"  gap length (lost frames): 1: {100*(gl==1).mean():.0f}%  2: {100*(gl==2).mean():.0f}%  3: {100*(gl==3).mean():.0f}%  4-6: {100*((gl>=4)&(gl<=6)).mean():.0f}%  >6: {100*(gl>6).mean():.0f}%")
for lo, hi, nm in ((1, 3, "short gaps (1-3 frames, repaired in 3D already)"), (4, 99, "long gaps (4+ frames)")):
    gc2 = Counter(); m = 0
    for p, f0, f1 in gaps:
        if not lo <= f1 - f0 - 1 <= hi: continue
        ks = [klass.get((p, f), "?") for f in range(f0 + 1, f1) if (p, f) in klass]
        if not ks: continue
        m += 1
        if all(k.startswith("R1") for k in ks): g = "all lost frames R1 (streak bridges both sides)"
        elif all(k.startswith(("R1", "R2")) for k in ks): g = "R1/R2 (streak, maybe one side)"
        elif any(k.startswith("R4") for k in ks): g = "a frame with fewer than 3 blobs"
        elif any(k.startswith("R3") for k in ks): g = "episode too long (R3)"
        else: g = "clean in 3 cameras (matcher/linker)"
        gc2[g] += 1
    print(f"  {nm}: {m}")
    for kk in sorted(gc2): print("     %-62s %6d  %5.1f%%" % (kk, gc2[kk], 100 * gc2[kk] / max(m, 1)))
