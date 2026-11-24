# ruff: noqa
"""Streaks on syn1: (1) is blob mass additive across merge/split events (truth identities),
(2) can truth-free 2D streaks + mass jumps flag merged blobs.   uv run python bench/limits/streaks_syn1.py [f0 f1]"""
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from openptv2.storage import RunStore

SYN = Path.home() / "Downloads/HiDImaging/Synthetic-CompleteTest"
F0, F1 = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (100, 160)
st = RunStore.open(SYN / "syn1" / "test", mode="r")
tr = np.load(SYN / "truth/truth.npz")

def auc(s, y):
    r = np.argsort(np.argsort(s)) + 1
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

# ---- load blobs and truth membership
B = {}      # (cam, f) -> rows
M = {}      # (cam, f) -> list of particle-id sets per blob
S = {}      # (cam, f) -> largest distance of a member's true pixel from the blob centre
for f in range(F0, F1 + 1):
    m = np.flatnonzero(tr["frame"] == f)
    for cam in range(4):
        rows = np.asarray(st.root[f"targets/cam_{cam}/frame_{f:06d}"]); B[cam, f] = rows
        px = tr["pixel_xy_per_cam"][m][:, 2 * cam: 2 * cam + 2]; ok = np.isfinite(px[:, 0])
        pid = tr["particle_id"][m][ok]; px = px[ok]
        d, j = cKDTree(rows[:, 1:3]).query(px, distance_upper_bound=3.0)
        mem = [[] for _ in rows]; sh = np.zeros(len(rows))
        for p, jj, dd in zip(pid, j, d):
            if np.isfinite(dd): mem[jj].append(int(p)); sh[jj] = max(sh[jj], dd)
        M[cam, f] = mem; S[cam, f] = sh

# ---- (1) mass additivity with truth identities
ev = defaultdict(list)    # kind -> log(mass ratio)
for cam in range(4):
    for f in range(F0, F1):
        own0 = {p: i for i, mm in enumerate(M[cam, f]) for p in mm}
        for c, mm in enumerate(M[cam, f + 1]):
            if not mm: continue
            par = {own0[p] for p in mm if p in own0}
            if not par: continue
            # parent blobs may hold other particles that went elsewhere: count only clean cases
            if any(set(M[cam, f][q]) - set(mm) for q in par) or len(par) == 0: continue
            kind = (len(mm), len(par))
            if kind in ((1, 1), (2, 2), (2, 1)):
                r = B[cam, f + 1][c, 6] / sum(B[cam, f][q, 6] for q in par)
                ev[kind].append(np.log(r))
print("mass ratio across one frame, log(child / sum of parents)   [truth identities, parents hold no one else]")
for k, nm in (((1, 1), "single -> single (no event)"), ((2, 2), "two parents -> one child with 2 (merge)"), ((2, 1), "one parent holding same 2 -> same 2")):
    v = np.array(ev[k]); 
    if len(v): print("  %-42s n=%6d  median %+.3f  5-95%% %+.3f..%+.3f  robust sd %.3f" % (nm, len(v), np.median(v), *np.percentile(v, [5, 95]), 1.4826 * np.median(np.abs(v - np.median(v)))))
# merge events by jump size: how much does a single-parent blob rise when a second particle arrives?
jump_in, jump_out = [], []
for cam in range(4):
    for f in range(F0, F1):
        for c, mm in enumerate(M[cam, f + 1]):
            pass
up, calm = [], []
for cam in range(4):
    for f in range(F0, F1):
        own1 = {p: i for i, mm in enumerate(M[cam, f + 1]) for p in mm}
        for a, mm in enumerate(M[cam, f]):
            if len(mm) != 1: continue
            p = mm[0]
            if p not in own1: continue
            c = own1[p]; k1 = len(M[cam, f + 1][c])
            r = np.log(B[cam, f + 1][c, 6] / B[cam, f][a, 6])
            (calm if k1 == 1 else up).append(r) if k1 in (1, 2) else None
up, calm = np.array(up), np.array(calm)
print("\nedge of a SINGLE blob at t to the blob holding the same particle at t+1:  log mass ratio")
print("  stays single   n=%d  median %+.3f  5-95%% %+.3f..%+.3f" % (len(calm), np.median(calm), *np.percentile(calm, [5, 95])))
print("  becomes merged n=%d  median %+.3f  5-95%% %+.3f..%+.3f   (ln2 = %.3f)" % (len(up), np.median(up), *np.percentile(up, [5, 95]), np.log(2)))
y = np.r_[np.zeros(len(calm)), np.ones(len(up))]
print("  AUC of |log ratio| for 'a second particle joined this edge':  %.3f" % auc(np.r_[np.abs(calm), np.abs(up)], y))
print("  AUC of  log ratio (signed):                                  %.3f" % auc(np.r_[calm, up], y))

# ================= (2) truth-free streaks (per camera) =================
GATE = 2.0
ISO = 3.5       # a blob with no other blob within ISO px is taken as single (anchor)

def link_cam(cam):
    """Hungarian frame to frame; returns tracklet id per (f, i) and the chains."""
    tid = {}; nxt = 0; prev = None
    for f in range(F0, F1 + 1):
        rows = B[cam, f]
        if prev is not None:
            pr, pt = prev
            d = np.hypot(pr[:, None, 1] - rows[None, :, 1], pr[:, None, 2] - rows[None, :, 2])
            c = d + 0.0 * np.abs(np.log(rows[None, :, 6] / pr[:, None, 6]))
            c = np.where(d <= GATE, c, 1e6)
            ri, ci = linear_sum_assignment(c)
            m = {int(j): int(i) for i, j in zip(ri, ci) if c[i, j] < 1e5}
        else:
            m = {}
        cur = []
        for j in range(len(rows)):
            if j in m: cur.append(pt[m[j]])
            else: cur.append(nxt); nxt += 1
        for j, t in enumerate(cur): tid[f, j] = t
        prev = (rows, cur)
    return tid

def run(cam, W, mode):
    """mass of each node / the unit mass of its tracklet, from nodes of the same tracklet within +-W frames.
    mode: 'q15' = 15th percentile of the tracklet's own mass (truth-free), 'oracle' = median over frames where truth says single."""
    tid = link_cam(cam)
    by = defaultdict(list)
    for (f, j), t in tid.items(): by[t].append((f, j))
    out = []     # (f, j, ratio, true m, tracklet length, has clean frame)
    for t, nodes in by.items():
        nodes.sort()
        fs = np.array([f for f, j in nodes]); ms = np.array([B[cam, f][j, 6] for f, j in nodes])
        ks = np.array([len(M[cam, f][j]) for f, j in nodes])
        for n, (f, j) in enumerate(nodes):
            if ks[n] == 0: continue
            w = np.abs(fs - f) <= W
            if mode == "q15":
                ref = np.percentile(ms[w], 15) if w.sum() >= 3 else np.nan
            else:
                sel = w & (ks == 1)
                ref = np.median(ms[sel]) if sel.any() else np.nan
            out.append((f, j, ms[n] / ref if np.isfinite(ref) else np.nan, ks[n], len(nodes), (ks == 1).any()))
    return np.array(out), tid

print("\n(2) streaks (per camera): flag a blob as merged when mass / (its streak's unit mass) is high")
for mode in ("oracle", "q15"):
    for W in (3, 6, 12, 30, 1000):
        a = np.vstack([run(c, W, mode)[0] for c in range(4)])
        ratio, k = a[:, 2], a[:, 3]; have = np.isfinite(ratio); ym = (k >= 2); y = ym[have].astype(int)
        def pr(t): 
            fl = ratio[have] > t
            return 100 * (fl & (y == 1)).sum() / y.sum(), 100 * (fl & (y == 1)).sum() / max(fl.sum(), 1)
        print("  %-6s +-%4d fr: merged with ref %3.0f%% | AUC %.3f | >1.4 recall/prec %2.0f/%2.0f%% | >1.25 %2.0f/%2.0f%%" % (mode, W, 100 * have[ym].mean(), auc(ratio[have], y), *pr(1.4), *pr(1.25)))
a = np.vstack([run(c, 1000, "q15")[0] for c in range(4)])
ym = a[:, 3] >= 2
print("  tracklet length (frames) of merged blobs: median %.0f, 10%% %.0f ; merged blobs on tracklets with NO clean frame: %.0f%%" % (np.median(a[ym, 4]), np.percentile(a[ym, 4], 10), 100 * (~a[ym, 5].astype(bool)).mean()))


# ================= (3) ablation: what each cue adds, scored on harm (centre shift) =================
def _fit_logit(X, y, l2=1.0, it=25):
    A = np.column_stack([np.ones(len(X)), X]); w = np.zeros(A.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-A @ w)); g = A.T @ (p - y) + l2 * np.r_[0, w[1:]]
        H = (A * (p * (1 - p))[:, None]).T @ A + l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
        w -= np.linalg.solve(H + 1e-9 * np.eye(len(w)), g)
    return w
def cross_val_logit(X, y, groups, k=5):
    out = np.zeros(len(y)); ug = np.unique(groups)
    for i in range(k):
        te = np.isin(groups, ug[i::k]); w = _fit_logit(X[~te], y[~te].astype(float))
        out[te] = 1 / (1 + np.exp(-(np.column_stack([np.ones(te.sum()), X[te]]) @ w)))
    return out
R_TOPO = 3.0; L_TOPO = 6
def features(cam, W=30):
    tid = link_cam(cam)
    by = defaultdict(list)
    for (f, j), t in tid.items(): by[t].append((f, j))
    for t in by: by[t].sort()
    first = {t: v[0] for t, v in by.items()}; last = {t: v[-1] for t, v in by.items()}
    # ends / starts by frame for topology
    ends = defaultdict(list); starts = defaultdict(list)
    for t, v in by.items():
        ends[v[-1][0]].append((v[-1][1], t)); starts[v[0][0]].append((v[0][1], t))
    rows = []
    for t, nodes in by.items():
        fs = np.array([f for f, j in nodes]); ms = np.array([B[cam, f][j, 6] for f, j in nodes]); ks = np.array([len(M[cam, f][j]) for f, j in nodes])
        for n, (f, j) in enumerate(nodes):
            if ks[n] == 0: continue
            r = B[cam, f]; x, y = r[j, 1], r[j, 2]
            w = np.abs(fs - f) <= W
            q = np.percentile(ms[w], 15) if w.sum() >= 3 else np.nan
            orc = np.median(ms[w & (ks == 1)]) if (w & (ks == 1)).any() else np.nan
            # topology: a tracklet ended in [f-L, f-1] next to this blob (position of its last node vs this tracklet's node at that frame)
            died = born = 99.0
            for g in range(max(F0, f - L_TOPO), f):
                for jj, tt in ends.get(g, []):
                    if tt == t: continue
                    k = np.flatnonzero(fs == g)
                    if len(k):
                        px_, py_ = B[cam, g][nodes[k[0]][1], 1:3]
                        died = min(died, np.hypot(B[cam, g][jj, 1] - px_, B[cam, g][jj, 2] - py_))
            for g in range(f + 1, min(F1, f + L_TOPO) + 1):
                for jj, tt in starts.get(g, []):
                    if tt == t: continue
                    k = np.flatnonzero(fs == g)
                    if len(k):
                        px_, py_ = B[cam, g][nodes[k[0]][1], 1:3]
                        born = min(born, np.hypot(B[cam, g][jj, 1] - px_, B[cam, g][jj, 2] - py_))
                    else:    # this tracklet ended before g: compare with its last position
                        if last[t][0] < g:
                            px_, py_ = B[cam, last[t][0]][last[t][1], 1:3]
                            born = min(born, np.hypot(B[cam, g][jj, 1] - px_, B[cam, g][jj, 2] - py_))
            nn = cKDTree(r[:, 1:3]).query(r[j, 1:3], k=2)[0][1]
            rows.append((cam * 10000 + f, r[j, 3], r[j, 6], ms[n] / q if np.isfinite(q) else np.nan, ms[n] / orc if np.isfinite(orc) else np.nan, died, born, nn, ks[n], S[cam, f][j], len(nodes), f))
    return np.array(rows)

print("\n(3) ablation, 5-fold by frame. target = blob centre more than 0.5 px from a member's true pixel ('harmful')")
D = np.vstack([features(c) for c in range(4)])
grp = D[:, 11].astype(int) // 4          # group by frame blocks
harm = D[:, 9] > 0.5; merged = D[:, 8] >= 2
print("  nodes %d | merged (m>=2) %.1f%% | harmful (>0.5 px) %.1f%%  | harmful that are merged %.0f%%" % (len(D), 100 * merged.mean(), 100 * harm.mean(), 100 * merged[harm].mean()))
def feat(cols):
    X = []
    for c in cols:
        if c == "n": X.append(np.log(D[:, 1]))
        elif c == "g": X.append(np.log(D[:, 2]))
        elif c == "nn": X.append(np.minimum(D[:, 7], 12))
        elif c == "qr": X.append(np.log(np.nan_to_num(D[:, 3], nan=1.0)))
        elif c == "or": X.append(np.log(np.nan_to_num(D[:, 4], nan=1.0)))
        elif c == "died": X.append(np.minimum(D[:, 5], 8))
        elif c == "born": X.append(np.minimum(D[:, 6], 8))
        elif c == "dieds": X.append(np.exp(-np.minimum(D[:, 5], 8) ** 2 / 4))
        elif c == "borns": X.append(np.exp(-np.minimum(D[:, 6], 8) ** 2 / 4))
    return np.column_stack(X)
def score(cols, y):
    X = feat(cols); X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    p = cross_val_logit(X, y, grp)
    # recall at 5% false alarms among non-targets
    thr = np.percentile(p[y == 0], 95)
    return auc(p, y.astype(int)), (p[y == 1] > thr).mean()
for tgt, y in (("merged (m>=2)", merged), ("harmful (>0.5 px)", harm)):
    print("  target:", tgt)
    for name, cols in (("blob alone (n, grey sum)", ["n", "g"]),
                       ("+ nearest neighbour in this frame", ["n", "g", "nn"]),
                       ("+ streak mass ratio (truth-free)", ["n", "g", "nn", "qr"]),
                       ("+ streak topology (ended/started next to it)", ["n", "g", "nn", "dieds", "borns"]),
                       ("+ both", ["n", "g", "nn", "qr", "dieds", "borns"]),
                       ("+ both, oracle unit mass", ["n", "g", "nn", "or", "dieds", "borns"])):
        a, rec = score(cols, y)
        print("    %-48s AUC %.3f   recall at 5%% false alarms %.0f%%" % (name, a, 100 * rec))
