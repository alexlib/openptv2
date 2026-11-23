import sys, numpy as np
from collections import Counter
from openptv2.storage import RunStore
from openptv2.recorrespond import load_parameters
from openptv2.correspondences import match_frame_correspondences
from openptv2.tracking_framebuf import TargetArray
B = "/Users/alex/Downloads/HiDImaging/tracker-bench-2026-09-29/cases/" + sys.argv[1] + "/"
mode = sys.argv[2] if len(sys.argv) > 2 else "oracle"
cpar, vpar, cals, _ = load_parameters(B + "parameters_Run1.yaml")
st = RunStore(B + "res/run.zarr", mode="r")
tr = np.load(B + "truth.npz")
frames = st.frames()[::10]
T = Counter()
for f in frames:
    raw = [np.asarray(st.root[f"targets/cam_{c}/frame_{f:06d}"]) for c in range(4)]
    dets = [st.read_targets(c, f) for c in range(4)]
    pos, ids = match_frame_correspondences(dets, cpar, cals, vpar)
    ids = np.asarray(ids)[:, :4]
    mall = np.flatnonzero(tr["frame"] == f)
    bt = tr["blob_tid"][mall]; nd = tr["ndet"][mall]
    owner = {}
    for k in range(len(mall)):
        for c in range(4):
            if bt[k, c] >= 0: owner.setdefault((c, int(bt[k, c])), []).append(k)
    nsh = np.array([sum(len(owner[(c, int(bt[k, c]))]) >= 2 for c in range(4) if bt[k, c] >= 0) for k in range(len(mall))])
    def judge(rows):
        found = set(); ghost = 0
        for r in rows:
            cnt = Counter()
            for c in range(4):
                if r[c] >= 0:
                    for k in owner.get((c, int(r[c])), []): cnt[k] += 1
            ncam = int((r >= 0).sum())
            if cnt and max(cnt.values()) >= 2 and max(cnt.values()) >= ncam - 1:
                found.add(max(cnt, key=cnt.get))
            else: ghost += 1
        return found, ghost
    f1, g1 = judge(ids)
    gain = [float(np.median(raw[c][:, 6])) for c in range(4)]
    used = [set(int(x) for x in ids[:, c] if x >= 0) for c in range(4)]
    # pass 2 detections: unused targets + blobs that are shared (oracle: truth; "bright": sumg > 1.4x camera median)
    sel = []
    for c in range(4):
        n = len(raw[c])
        if mode.startswith("xcam"):
            thr = float(mode.split(":")[1])
            sh = set()
            for r in ids:
                cams = [cc for cc in range(4) if r[cc] >= 0]
                if len(cams) < 3: continue
                b = np.array([raw[cc][r[cc], 6] / gain[cc] for cc in cams])
                ref = np.sort(b)[1]
                if cams and c in cams and b[cams.index(c)] > thr * ref: sh.add(int(r[c]))
        elif mode == "oracle":
            sh = {b for (cc, b), ks in owner.items() if cc == c and len(ks) >= 2}
        else:
            med = np.median(raw[c][:, 6]); sh = set(np.flatnonzero(raw[c][:, 6] > float(mode) * med).tolist())
        keep = [i for i in range(n) if (i not in used[c]) or (i in sh)]
        sel.append(keep)
    d2 = []
    for c in range(4):
        keep = sorted(sel[c], key=lambda i: raw[c][i, 2])
        ta = TargetArray(len(keep))
        for j, i in enumerate(keep):
            r = raw[c][i]
            ta[j].set_pnr(j); ta[j].set_pos((float(r[1]), float(r[2])))
            ta[j].set_pixel_counts(int(r[3]), int(r[4]), int(r[5])); ta[j].set_sum_grey_value(int(max(r[6], 1))); ta[j].set_tnr(-1)
        d2.append((ta, keep))
    pos2, ids2 = match_frame_correspondences([x[0] for x in d2], cpar, cals, vpar)
    ids2 = np.asarray(ids2)[:, :4]
    new = []
    for r in ids2:
        row = np.array([d2[c][1][r[c]] if r[c] >= 0 else -1 for c in range(4)])
        has_unused = any(row[c] >= 0 and row[c] not in used[c] for c in range(4))
        if has_unused and (row >= 0).sum() >= 3: new.append(row)
    f2, g2 = judge(new) if new else (set(), 0)
    gain = f2 - f1
    T["rows1"] += len(ids); T["ghost1"] += g1; T["new_rows"] += len(new); T["new_ghost"] += g2; T["new_correct_particles"] += len(gain)
    miss = (nd >= 2) & (nsh >= 2) & np.array([k not in f1 for k in range(len(mall))])
    T["missing_shared2"] += int(miss.sum()); T["recovered"] += int(sum(1 for k in gain if miss[k]))
print(f"mode {mode}: pass-1 rows {T['rows1']} (ghost {T['ghost1']})  | pass-2 new rows {T['new_rows']}: correct new particles {T['new_correct_particles']}, wrong rows {T['new_ghost']} (precision {1-T['new_ghost']/max(T['new_rows'],1):.3f})")
print(f"   missing particles with >=2 shared cams: {T['missing_shared2']}  recovered {T['recovered']} ({T['recovered']/T['missing_shared2']:.3f})")
