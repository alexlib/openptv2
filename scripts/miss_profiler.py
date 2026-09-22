"""Miss profiler + appearance probe on wp1 (offline, existing outputs).

1. Miss profiler: for every reference link (matched vs missed by our
   fwd+bwd+post run), tabulate step, 3-pt acc, local crowding (neighbors
   within 1.9 mm of the target), history length, frame. Shows WHERE the
   headroom is.
2. Appearance probe: at contested divergences (ref a->b missed, test
   a->c taken), compare brightness/size continuity |sumg(b)-sumg(a)| vs
   |sumg(c)-sumg(a)| across cameras. Systematic truth-ward signal =>
   appearance belongs in the candidate cost.

Run from repo root (needs scratch/_wp1_ab/res + dataset targets):
    uv run python -u scripts/miss_profiler.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DS = Path(r"C:\Users\alex\Downloads\HiDImaging\wp1_10_images")
REF = DS / "res_ground_truth_backup"
TST = REPO / "scratch" / "_wp1_ab" / "res"
FIRST, LAST = 100001, 100010


def read_pn(d: Path, f: int):
    l = (d / f"ptv_is.{f}").read_text().splitlines()
    n = int(l[0])
    p, x = [], []
    for line in l[1: n + 1]:
        s = line.split()
        p.append(int(s[0]))
        x.append(int(s[1]))
    return p, x


def read_xyz(f: int):
    l = (REF / f"rt_is.{f}").read_text().splitlines()
    n = int(l[0])
    pos = np.array([[float(v) for v in s.split()[1:4]] for s in l[1: n + 1]])
    return pos, n


def load_sumg():
    """(frame, cam, target_row) -> (sumg, n_px). Keeps x,y too."""
    out = {}
    for f in range(FIRST, LAST + 1):
        for cam in range(1, 5):
            tl = (REPO / "scratch" / "_wp1_ab" / "img_3dptv"
                  / f"Cam{cam}.{f}_targets").read_text().splitlines()
            for line in tl[1:]:
                s = line.split()
                out[(f, cam, int(s[0]))] = (float(s[6]), int(s[3]))
    return out


def main():
    XYZ = {f: read_xyz(f) for f in range(FIRST, LAST + 1)}
    RP, RN = {}, {}
    for f in range(FIRST, LAST + 1):
        RP[f], RN[f] = read_pn(REF, f)
    TP, TN = {}, {}
    for f in range(FIRST, LAST + 1):
        TP[f], TN[f] = read_pn(TST, f)

    # ref chains for history length
    node2chain = {}
    for f in range(FIRST, LAST + 1):
        for i in range(len(RP[f])):
            if RP[f][i] < 0:
                ch, (cf, ci) = [(f, i)], (f, i)
                while cf <= LAST and ci < len(RN[cf]) and RN[cf][ci] >= 0:
                    ci = RN[cf][ci]
                    cf += 1
                    ch.append((cf, ci))
                for k, node in enumerate(ch):
                    node2chain[node] = (len(ch), k)

    from scipy.spatial import cKDTree
    trees = {f: cKDTree(XYZ[f][0]) for f in XYZ}

    feats = {"hit": [], "miss": []}
    contests = []  # (acc_ref, acc_test, app_ref, app_test)
    SUMG = load_sumg()

    def particle_app(f, row):
        """Mean sumg over correspondent cameras of rt row."""
        line = (REF / f"rt_is.{f}").read_text().splitlines()[row + 1].split()
        vals = []
        for cam in range(1, 5):
            j = int(line[3 + cam])
            if j >= 0 and (f, cam, j) in SUMG:
                vals.append(SUMG[(f, cam, j)][0])
        return np.mean(vals) if vals else float("nan")

    for f in range(FIRST + 1, LAST + 1):
        t = TP[f]
        X1, _ = XYZ[f - 1]
        X3, _ = XYZ[f]
        for i, p in enumerate(RP[f]):
            if p < 0 or p >= len(X1) or i >= len(X3):
                continue
            step = float(np.linalg.norm(X3[i] - X1[p]))
            crowd = len(trees[f].query_ball_point(X3[i], r=1.9)) - 1
            L, k = node2chain.get((f - 1, p), (1, 0))
            pp = RP[f - 1][p] if p < len(RP[f - 1]) else -1
            if 0 <= pp:
                X0, _ = XYZ[f - 2]
                acc = float(np.linalg.norm(X3[i] - 2 * X1[p] + X0[pp])) \
                    if pp < len(X0) else float("nan")
            else:
                acc = float("nan")
            hit = i < len(t) and t[i] == p
            feats["hit" if hit else "miss"].append((step, acc, crowd, L, f))
            if not hit:
                # test's NEXT of p: where did test think p goes?
                cn = TN[f - 1][p] if p < len(TN[f - 1]) else -2
                if cn >= 0 and cn < len(X3) and cn != i:
                    c = cn
                    a_ref, a_tst = acc, float("nan")
                    if 0 <= pp and pp < len(XYZ[f - 2][0]):
                        a_tst = float(np.linalg.norm(
                            X3[c] - 2 * X1[p] + XYZ[f - 2][0][pp]))
                    sa = particle_app(f - 1, p)
                    sb = particle_app(f, i)
                    sc = particle_app(f, c)
                    contests.append((a_ref, a_tst, abs(sb - sa), abs(sc - sa)))

    print("== miss profiler (matched vs missed ref links) ==")
    for name in ("hit", "miss"):
        A = np.array([r[0] for r in feats[name]])
        C = np.array([r[2] for r in feats[name]])
        L = np.array([r[3] for r in feats[name]])
        accs = np.array([r[1] for r in feats[name]])
        accs = accs[~np.isnan(accs)]
        print(f"{name:5} n={len(A):5} step_mean={A.mean():.3f} "
              f"step_p90={np.percentile(A, 90):.3f} "
              f"crowd_mean={C.mean():.2f} crowd>=2:{(C >= 2).mean():.1%} "
              f"histlen_mean={L.mean():.2f} starts:{(L == 1).mean():.1%} "
              f"acc_mean={accs.mean():.3f} acc_p90={np.percentile(accs, 90):.3f}")
    print("\nmiss frame histogram:",
          {f: sum(1 for r in feats['miss'] if r[4] == f)
           for f in range(FIRST + 1, LAST + 1)})

    C = np.array(contests)
    C = C[~np.isnan(C).any(axis=1)]
    print(f"\n== contested divergences with full data: {len(C)} ==")
    if len(C):
        print(f"acc: ref-choice mean={C[:, 0].mean():.3f} vs "
              f"test-choice mean={C[:, 1].mean():.3f}; "
              f"test smoother {(C[:, 1] < C[:, 0]).mean():.1%}")
        print(f"appearance |dsumg|: ref-choice mean={C[:, 2].mean():.1f} vs "
              f"test-choice mean={C[:, 3].mean():.1f}; "
              f"truth more similar {(C[:, 2] < C[:, 3]).mean():.1%}")


if __name__ == "__main__":
    main()
