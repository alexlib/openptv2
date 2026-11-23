import sys, numpy as np
from collections import Counter
from openptv2.storage import RunStore
from openptv2.recorrespond import load_parameters
from openptv2.correspondences import match_frame_correspondences
B = "/Users/alex/Downloads/HiDImaging/tracker-bench-2026-09-29/cases/" + sys.argv[1] + "/"
cpar, vpar, cals, _ = load_parameters(B + "parameters_Run1.yaml")
st = RunStore(B + "res/run.zarr", mode="r")
tr = np.load(B + "truth.npz")
frames = st.frames()[::10]
def score(pair_flag):
    cpar.pair_flag = pair_flag
    tot = Counter()
    for f in frames:
        dets = [st.read_targets(c, f) for c in range(4)]
        pos, ids = match_frame_correspondences(dets, cpar, cals, vpar)
        ids = np.asarray(ids)[:, :4]
        mall = np.flatnonzero(tr["frame"] == f)
        bt = tr["blob_tid"][mall]; nd = tr["ndet"][mall]
        owner = {}
        for k in range(len(mall)):
            for c in range(4):
                if bt[k, c] >= 0: owner.setdefault((c, int(bt[k, c])), []).append(k)
        # shared count per particle
        nsh = np.array([sum(len(owner[(c, int(bt[k, c]))]) >= 2 for c in range(4) if bt[k, c] >= 0) for k in range(len(mall))])
        found = np.zeros(len(mall), bool)
        good = ghost = 0
        for r in range(len(ids)):
            cnt = Counter()
            for c in range(4):
                if ids[r, c] >= 0:
                    for k in owner.get((c, int(ids[r, c])), []): cnt[k] += 1
            ncam = int((ids[r] >= 0).sum())
            if cnt and max(cnt.values()) >= 2 and max(cnt.values()) >= ncam - 1:
                k = max(cnt, key=cnt.get); found[k] = True; good += 1
            else:
                ghost += 1
        tot["rows"] += len(ids); tot["good"] += good; tot["ghost"] += ghost
        for lab, m in (("clean", (nd >= 2) & (nsh == 0)), ("shared", (nd >= 2) & (nsh > 0)), ("shared>=2cams", (nd >= 2) & (nsh >= 2))):
            tot[lab + "_n"] += int(m.sum()); tot[lab + "_found"] += int((found & m).sum())
    return tot
for pf in (0, 1):
    t = score(pf)
    print(f"pair_flag={pf}: rows {t['rows']}  correct {t['good']}  ghost rows {t['ghost']} ({t['ghost']/t['rows']:.3f})  | found: clean {t['clean_found']/t['clean_n']:.3f}  shared {t['shared_found']/t['shared_n']:.3f}  shared>=2cams {t['shared>=2cams_found']/t['shared>=2cams_n']:.3f}")
