import sys, numpy as np
sys.path.insert(0, "scripts")
from scipy.spatial import cKDTree
import synth_bench as sb
B = "/Users/alex/Downloads/HiDImaging/tracker-bench-2026-09-29/"
case = sys.argv[1]; names = sys.argv[2:]
tr = np.load(B + f"cases/{case}/truth.npz")
tid, tfr, tpos, nd, bt = tr["particle_id"], tr["frame"], tr["pos_mm"], tr["ndet"], tr["blob_tid"]
n = len(tid)
# sharing: truth points in the same frame with the same blob in a camera
share = np.zeros((n, 4), int)
for c in range(4):
    m = np.flatnonzero(bt[:, c] >= 0)
    key = tfr[m].astype(np.int64) * 10_000_000 + bt[m, c]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    share[m, c] = cnt[inv]
nclean = ((share == 1)).sum(1)
anyshared = (share >= 2).any(1)
by_frame = {}
for f in np.unique(tfr):
    m = tfr == f
    by_frame[int(f)] = (tpos[m], tid[m], cKDTree(tpos[m]))
idx_of = {}
for i in range(n): idx_of[(int(tid[i]), int(tfr[i]))] = i
res = {}
for nm in names:
    p = np.load(B + f"runs/{case}/{nm}/pred.npz")
    fr, pos = p["frame"], p["pos"]
    g0, _, off = sb._match(by_frame, fr, pos, 0.6)
    bias = np.median(off[g0 >= 0], axis=0)
    pos = pos - bias
    gid, dist, _ = sb._match(by_frame, fr, pos, 0.4)
    ok = gid >= 0
    found = np.zeros(n, bool); err = np.full(n, np.nan)
    for g, f, d in zip(gid[ok], fr[ok], dist[ok]):
        i = idx_of[(int(g), int(f))]; found[i] = True; err[i] = d
    # unmatched points near a truth (misplaced) : within 1.0 mm
    res[nm] = (found, err, (~ok).sum(), len(fr))
cats = {
 "ndet>=2, all clean (share==1 in all detected)": (nd >= 2) & ~anyshared,
 "ndet>=2, some shared blob": (nd >= 2) & anyshared,
 "ndet>=2, clean cams<2 (needs split)": (nd >= 2) & (nclean < 2),
 "ndet<2 (no stereo)": nd < 2,
}
print(f"{'category':48s} {'n':>7s} " + " ".join(f"{nm[:22]:>22s}" for nm in names))
for cn, m in cats.items():
    row = f"{cn:48s} {m.sum():7d} "
    for nm in names:
        f, e, _, _ = res[nm]
        row += f"{f[m].mean():8.3f} err {np.nanmean(e[m & f]):6.3f}  "
    print(row)
print("total yield", [round(res[nm][0].mean(), 3) for nm in names], " unmatched points", [res[nm][2] for nm in names])
# what holistic recovers vs two_phase
a, b = res[names[0]][0], res[names[-1]][0]
rec, lost = b & ~a, a & ~b
print("recovered", rec.sum(), "lost", lost.sum())
for cn, m in cats.items():
    print(f"  {cn:48s} recovered {int((rec & m).sum()):6d} lost {int((lost & m).sum()):6d}")
# position error on common set by sharing
both = a & b
for lab, m in (("all clean", ~anyshared), ("some shared", anyshared)):
    mm = both & m
    print(f"common set {lab:12s} n={mm.sum():6d} err {names[0][:12]} {np.nanmean(res[names[0]][1][mm]):.3f} -> {np.nanmean(res[names[-1]][1][mm]):.3f}")
print()
print("by (number of cameras where the particle is detected / in a shared blob):  rows = ndet, cols = n shared cams")
miss = ~a & ~b & (nd >= 2)
nsh = (share >= 2).sum(1)
for lab, m in (("two_phase has it", a), ("recovered by holistic", rec), ("missed by both", miss)):
    print(lab, int(m.sum()))
    for d in (2, 3, 4):
        print("  ndet", d, [int((m & (nd == d) & (nsh == k)).sum()) for k in range(0, d + 1)])
