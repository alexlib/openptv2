"""Manual-vote evidence pack for the 34 union-miss links.

For each: the reference option (p->i) vs the engines' top option (q->i,
same history) with step/acc/appearance on both sides, onward consistency
(ref-next of p and q, test-next of p and q), and a verdict hint:
  REF-PLAUSIBLE  ref option smooth/uncrowded, engines stole from a
                 live track (swap)
  ENG-PLAUSIBLE  engines' option smoother, ref jumps far/fast
  REF-FAST       ref continues established fast motion (low acc, big step)
  AMBIGUOUS      both defensible
  TAIL           last-frame boundary (no lookahead for anyone)

Run from repo root:
    uv run python -u scripts/judge_hard_links.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DS = Path(r"C:\Users\alex\Downloads\HiDImaging\wp1_10_images")
REF = DS / "res_ground_truth_backup"
ENGS = {
    "parity": REPO / "scratch" / "_wp1_ab" / "res",
    "default": REPO / "scratch" / "_ens_lr1" / "res",
    "4be": REPO / "scratch" / "_ens_4be" / "res",
    "2p": REPO / "scratch" / "_ens_2p" / "res_2p",
}


def read_pn(d: Path, f: int):
    lines = (d / f"ptv_is.{f}").read_text().splitlines()
    n = int(lines[0])
    p, x = [], []
    for line in lines[1: n + 1]:
        s = line.split()
        p.append(int(s[0]))
        x.append(int(s[1]))
    return p, x


def read_xyz(f: int):
    lines = (REF / f"rt_is.{f}").read_text().splitlines()
    n = int(lines[0])
    return np.array([[float(v) for v in l.split()[1:4]] for l in lines[1: n + 1]])


def load_sumg():
    out = {}
    for f in range(100001, 100011):
        for cam in range(1, 5):
            tl = (REPO / "scratch" / "_wp1_ab" / "img_3dptv"
                  / f"Cam{cam}.{f}_targets").read_text().splitlines()
            for line in tl[1:]:
                s = line.split()
                out[(f, cam, int(s[0]))] = float(s[6])
    return out


RT_COLS = {}


def rt_pcols(f: int):
    if f not in RT_COLS:
        lines = (REF / f"rt_is.{f}").read_text().splitlines()
        RT_COLS[f] = [l.split()[4:8] for l in lines[1:]]
    return RT_COLS[f]


def app_of(f, row, SUMG):
    vals = []
    pc = rt_pcols(f)
    if row >= len(pc):
        return float("nan")
    for cam in range(1, 5):
        j = pc[row][cam - 1]
        if j != "-1" and (f, cam, int(j)) in SUMG:
            vals.append(SUMG[(f, cam, int(j))])
    return np.mean(vals) if vals else float("nan")


def main():
    from collections import Counter
    names = list(ENGS)
    P = {n: {f: read_pn(d, f)[0] for f in range(100002, 100011)}
         for n, d in ENGS.items()}
    N = {n: {f: read_pn(d, f)[1] for f in range(100002, 100011)}
         for n, d in ENGS.items()}
    RP = {f: read_pn(REF, f)[0] for f in range(100001, 100011)}
    RN = {f: read_pn(REF, f)[1] for f in range(100001, 100011)}
    XYZ = {f: read_xyz(f) for f in range(100001, 100011)}
    SUMG = load_sumg()

    print(f"{'frame':>8}{'row':>6}{'ref':>6}{'eng':>6}"
          f"{'stepR':>7}{'accR':>7}{'stepE':>7}{'accE':>7}"
          f"{'appR':>7}{'appE':>7}{'qRefNxt':>8}{'pTstNxt':>8}  verdict",
          flush=True)
    verdicts: Counter = Counter()
    for f in range(100003, 100011):
        r = RP[f]
        n = min(len(r), *(len(P[x][f]) for x in names))
        for i in range(n):
            if r[i] < 0:
                continue
            votes = [P[x][f][i] for x in names]
            if any(v == r[i] for v in votes):
                continue
            p = r[i]
            cnt = Counter(votes)
            top, c = cnt.most_common(1)[0]
            q = top if top is not None and top >= 0 else -1
            X1, X3 = XYZ[f - 1], XYZ[f]
            # Options share the SAME target i, differ in source:
            #   R (ref):      p -> i      E (engines' top): q -> i
            stepR = float(np.linalg.norm(X3[i] - X1[p])) \
                if 0 <= p < len(X1) else -1
            pp = RP[f - 1][p] if 0 <= p < len(RP[f - 1]) else -1
            if 0 <= pp and pp < len(XYZ[f - 2]):
                accR = float(np.linalg.norm(X3[i] - 2 * X1[p] + XYZ[f - 2][pp]))
            else:
                accR = float("nan")
            stepE = float(np.linalg.norm(X3[i] - X1[q])) \
                if 0 <= q < len(X1) else -1
            gq = RP[f - 1][q] if 0 <= q < len(RP[f - 1]) else -1
            if 0 <= gq and gq < len(XYZ[f - 2]):
                accE = float(np.linalg.norm(X3[i] - 2 * X1[q] + XYZ[f - 2][gq]))
            else:
                accE = float("nan")
            sa = app_of(f - 1, p, SUMG)
            appR = abs(app_of(f, i, SUMG) - sa)
            appE = abs(app_of(f, i, SUMG) - app_of(f - 1, q, SUMG)) \
                if q >= 0 else float("nan")
            qrefnxt = RN[f - 1][q] if 0 <= q < len(RN[f - 1]) else -9
            ptestnxt = N["default"][f - 1][p] \
                if 0 <= p < len(N["default"][f - 1]) else -9
            if f == 100010:
                v = "TAIL"
            elif not np.isnan(accR) and accR < 0.6 and stepR > 1.5:
                v = "REF-FAST"
            elif not np.isnan(accE) and not np.isnan(accR) and accE + 0.3 < accR:
                v = "ENG-PLAUSIBLE"
            elif not np.isnan(accE) and not np.isnan(accR) and accR + 0.3 < accE:
                v = "REF-PLAUSIBLE"
            else:
                v = "AMBIGUOUS"
            verdicts[v] += 1
            print(f"{f:>8}{i:>6}{p:>6}{q:>6}"
                  f"{stepR:>7.2f}{accR:>7.2f}{stepE:>7.2f}{accE:>7.2f}"
                  f"{appR:>7.0f}{appE:>7.0f}{qrefnxt:>8}{ptestnxt:>8}  {v}",
                  flush=True)
    print(f"\nverdicts: {dict(verdicts)}", flush=True)


if __name__ == "__main__":
    main()
