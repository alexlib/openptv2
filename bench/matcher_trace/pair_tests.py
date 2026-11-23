# ruff: noqa  (one-off analysis script)
import sys, numpy as np
from collections import Counter
from openptv2.storage import RunStore
from openptv2.recorrespond import load_parameters
from openptv2.loco import pixels_to_flat
from openptv2.algorithms.epi import epi_mm_batch
B = "/Users/alex/Downloads/HiDImaging/tracker-bench-2026-09-29/cases/" + sys.argv[1] + "/"
cpar, vpar, cals, _ = load_parameters(B + "parameters_Run1.yaml")
st = RunStore(B + "res/run.zarr", mode="r")
tr = np.load(B + "truth.npz")
frames = st.frames()[::7]
eps, cn, cnx, cny, csumg = vpar.eps0, vpar.cn, vpar.cnx, vpar.cny, vpar.csumg
print("eps0 %.3f mm, cn %.2f cnx %.2f cny %.2f csumg %.2f" % (eps, cn, cnx, cny, csumg), " pix_x", cpar.pix_x)
rows = []
parts = []   # one row per particle
for f in frames:
    T = [np.asarray(st.root[f"targets/cam_{c}/frame_{f:06d}"]) for c in range(4)]
    flat = [pixels_to_flat(T[c][:, 1:3], cals[c], cpar) for c in range(4)]
    pos, ids = st.read_correspondences(f)
    m = np.flatnonzero((tr["frame"] == f) & (tr["ndet"] >= 2))
    bt = tr["blob_tid"][m]
    # share count per blob (all truth, including ndet<2)
    mall = np.flatnonzero(tr["frame"] == f)
    shr = {}
    for c in range(4):
        cnt = Counter(tr["blob_tid"][mall, c][tr["blob_tid"][mall, c] >= 0].tolist())
        shr[c] = cnt
    # represented: a matcher row sharing >=2 blob ids with the truth particle
    key = {}
    for r in range(len(ids)):
        for c in range(4):
            if ids[r, c] >= 0: key.setdefault((c, int(ids[r, c])), []).append(r)
    for k, i in enumerate(m):
        have = [c for c in range(4) if bt[k, c] >= 0]
        cand = Counter()
        for c in have:
            for r in key.get((c, int(bt[k, c])), []): cand[r] += 1
        rep = bool(cand) and max(cand.values()) >= 2
        nsh = sum(shr[c][int(bt[k, c])] >= 2 for c in have)
        P = np.zeros((4, 4), bool)
        for a in have:
            for b in have:
                if a == b: continue
                ta, tb = T[a][bt[k, a]], T[b][bt[k, b]]
                xa_, ya_ = flat[a][bt[k, a]]; xb_, yb_ = flat[b][bt[k, b]]
                xmin, ymin, xmax, ymax = [float(v[0]) for v in epi_mm_batch(np.array([xa_]), np.array([ya_]), cals[a], cals[b], cpar.mm, vpar)]
                xs, ys = (xmin, xmax), (ymin, ymax)
                if xmin == xmax: xmax += 1e-10
                mm = (ymax - ymin) / (xmax - xmin); bb = ymin - mm * xmin
                d = abs((yb_ - mm * xb_ - bb) / np.sqrt(mm * mm + 1))
                lo_x, hi_x = min(xmin, xmax), max(xmin, xmax); lo_y, hi_y = min(ymin, ymax), max(ymin, ymax)
                inbox = (lo_x - eps < xb_ < hi_x + eps) and (lo_y - eps < yb_ < hi_y + eps)
                q = lambda u, v: (u / v if u < v else v / u)
                qn, qnx, qny, qs = q(ta[3], tb[3]), q(ta[4], tb[4]), q(ta[5], tb[5]), q(ta[6], tb[6])
                sa, sb = shr[a][int(bt[k, a])] >= 2, shr[b][int(bt[k, b])] >= 2
                rows.append((rep, nsh, len(have), sa, sb, d, inbox, qn, qnx, qny, qs))
                P[a, b] = (d < eps) and inbox and qn >= cn and qnx >= cnx and qny >= cny and qs > csumg
        import itertools
        Pm = P & P.T       # an edge exists if it passes from either side? the matcher uses the directed edge a->b for a<b
        Pu = np.triu(P, 1)
        best = 1
        for sz in (4, 3):
            if any(all(Pu[x, y] for x, y in itertools.combinations(cs, 2)) for cs in itertools.combinations(have, sz) if len(cs) == sz):
                best = sz; break
        if best == 1 and any(Pu[x, y] for x, y in itertools.combinations(have, 2)): best = 2
        used_by_other = 0
        for c in have:
            rr_ = key.get((c, int(bt[k, c])), [])
            if rr_ and not (rep and max(cand.values()) >= 2 and all(r in [q for q, v in cand.items() if v >= 2] for r in rr_)):
                used_by_other += 1
        parts.append((rep, nsh, len(have), best, used_by_other))
R = np.array(rows, float)
np.save("/private/tmp/claude-501/trace_rows.npy", R)
np.save("/private/tmp/claude-501/trace_parts.npy", np.array(parts, float))
print("pairs", len(R))
