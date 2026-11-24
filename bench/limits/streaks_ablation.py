# ruff: noqa
"""Do streaks (blob chains in one camera over time) help to find the blobs that hurt, on syn1 (truth) and real wp2 (LOCO)?

    uv run python bench/limits/streaks_ablation.py <label> <dir with res/run.zarr + parameters_Run1.yaml> <f0> <f1> [truth]

Labels of harm:
  truth   (syn1 only) blob centre more than 0.5 px from a member's true pixel
  loco    leave-one-camera-out residual of the blob's own 3D point above 0.5 px (needs a 4-camera point)
  loco90  same, but above the 90th percentile of this dataset
Also prints how the streak mass jumps look (calm edges) so real and synthetic can be compared.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
import streak_lib as sl
from openptv2.loco import loco_residuals
from openptv2.recorrespond import load_parameters
from openptv2.storage import RunStore

label, root, F0, F1 = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
use_truth = len(sys.argv) > 5 and sys.argv[5] == "truth"
st = RunStore.open(root, mode="r")
frames = [f for f in range(F0, F1 + 1)]
B = {(c, f): np.asarray(st.root[f"targets/cam_{c}/frame_{f:06d}"]) for f in frames for c in range(4)}
print(f"[{label}] frames {F0}-{F1}, blobs per camera and frame: {np.mean([len(v) for v in B.values()]):.0f}")

D = np.vstack([sl.node_features(B, c, F0, F1) for c in range(4)])
key = {(int(c), int(f), int(j)): i for i, (c, f, j) in enumerate(D[:, :3])}
grp = D[:, 1].astype(int) // 4

# --- calm edges: how much does the mass of a 1-1 linked blob change from one frame to the next
ms = {}
for (c, f), r in B.items():
    ms[c, f] = r[:, 6]
lr = []
for c in range(4):
    tid = sl.link_cam(B, c, F0, F1)
    for f in range(F0, F1):
        for j in range(len(B[c, f])):
            pass
    inv = {}
    for (f, j), t in tid.items():
        inv.setdefault((f, t), j)
    for (f, j), t in tid.items():
        if f < F1 and (f + 1, t) in inv:
            lr.append(np.log(ms[c, f + 1][inv[f + 1, t]] / ms[c, f][j]))
lr = np.array(lr)
med = np.median(lr)
print(f"[{label}] mass change of a linked blob per frame: robust sd {1.4826 * np.median(np.abs(lr - med)):.3f}, share above x1.3 (up) {100 * (lr > np.log(1.3)).mean():.1f}%, below /1.3 {100 * (lr < -np.log(1.3)).mean():.1f}%")

# --- LOCO residual per blob
cpar, vpar, cals, _ = load_parameters(root / "parameters_Run1.yaml")
loco = np.full(len(D), np.nan)
for f in frames:
    pos, ids = st.read_correspondences(f)
    ids = np.asarray(ids)[:, :4]
    full = (ids >= 0).all(1)
    if not full.any():
        continue
    px = np.full((int(full.sum()), 4, 2), np.nan)
    for c in range(4):
        px[:, c] = B[c, f][ids[full, c], 1:3]
    r = loco_residuals(px, cpar, cals, vpar)
    for c in range(4):
        for j, v in zip(ids[full, c], r[:, c]):
            i = key.get((c, f, int(j)))
            if i is not None and np.isfinite(v):
                loco[i] = v
has = np.isfinite(loco)
print(f"[{label}] blobs with a LOCO residual: {has.sum()} ({100 * has.mean():.0f}%), median {np.median(loco[has]):.3f}px, p90 {np.percentile(loco[has], 90):.3f}px, share above 0.5px {100 * (loco[has] > 0.5).mean():.1f}%")

# geometric merge label: a 3D point seen by 3 cameras projects within 1.5 px of a blob of the 4th camera -> that blob is shared
from openptv2.loco import project_pixels
geo = np.zeros(len(D), bool); geo_known = np.ones(len(D), bool)
for f in frames:
    pos, ids = st.read_correspondences(f)
    ids = np.asarray(ids)[:, :4]; pos = np.asarray(pos, float)
    for c in range(4):
        miss = (ids[:, c] < 0) & ((ids >= 0).sum(1) == 3)
        if not miss.any() or len(B[c, f]) == 0:
            continue
        pr = project_pixels(pos[miss], cals[c], cpar)
        d, k = cKDTree(B[c, f][:, 1:3]).query(pr, distance_upper_bound=1.5)
        for dd, kk in zip(d, k):
            if np.isfinite(dd):
                i = key.get((c, f, int(kk)))
                if i is not None: geo[i] = True
print(f"[{label}] geometric merge label (a 3-camera point projects onto this blob): {100 * geo.mean():.1f}% of blobs")
targets = {"geo-merged (3-cam point inside)": (np.ones(len(D), bool), geo), "loco>0.5px": (has, loco[has] > 0.5), "loco>p90": (has, loco[has] > np.percentile(loco[has], 90))}

if use_truth:
    tr = np.load(root.parent.parent / "truth/truth.npz")
    shift = np.zeros(len(D)); cnt = np.zeros(len(D))
    for f in frames:
        m = np.flatnonzero(tr["frame"] == f)
        for c in range(4):
            px = tr["pixel_xy_per_cam"][m][:, 2 * c: 2 * c + 2]
            ok = np.isfinite(px[:, 0]); px = px[ok]
            d, j = cKDTree(B[c, f][:, 1:3]).query(px, distance_upper_bound=3.0)
            for dd, jj in zip(d, j):
                if np.isfinite(dd):
                    i = key[c, f, int(jj)]
                    shift[i] = max(shift[i], dd); cnt[i] += 1
    keep = cnt > 0
    gk = np.flatnonzero(keep)
    print(f"[{label}] geometric label vs truth: of blobs labelled geo-merged, {100 * (cnt[geo & keep] >= 2).mean():.0f}% truly hold >=2 particles; recall of truly merged {100 * geo[keep & (cnt >= 2)].mean():.0f}%")
    targets = {"truth: merged (m>=2)": (keep, cnt[keep] >= 2), "truth: shift>0.5px": (keep, shift[keep] > 0.5), **targets}
    # does LOCO see the truth harm?
    both = keep & has
    print(f"[{label}] LOCO residual vs truth shift: correlation {np.corrcoef(loco[both], shift[both])[0, 1]:.2f}; AUC of LOCO for truth shift>0.5px: {sl.auc(loco[both], (shift[both] > 0.5).astype(int)):.3f}")

for name, (sel, y) in targets.items():
    Ds = D[sel]
    print(f"[{label}] target {name}: {len(Ds)} blobs, {100 * y.mean():.1f}% positive")
    for nm, a, rec in sl.ablate(Ds, y, grp[sel], sl.SETS):
        print(f"      {nm:50s} AUC {a:.3f}   recall at 5% false alarms {100 * rec:.0f}%")
