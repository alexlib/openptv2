"""Single-trajectory comparison: what does the residual 2% LOOK like?

Finds the most illustrative divergent trajectories (mid-splits, merges,
lost chains, ghost extensions), prints frame-by-frame tables (positions,
step, acc for each engine's choice) and renders ref-vs-test overlay plots.

Run from repo root (uses existing outputs, no tracking run):
    uv run python -u scripts/compare_single_trajectories.py \
        --work scratch/_wp1_ab --out scratch/single_traj
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DS = Path(r"C:\Users\alex\Downloads\HiDImaging\wp1_10_images")
REF = DS / "res_ground_truth_backup"
FIRST, LAST = 100001, 100010


def read_pn(res: Path, frame: int):
    lines = (res / f"ptv_is.{frame}").read_text().strip().splitlines()
    n = int(lines[0])
    prev, nxt = [], []
    for line in lines[1: n + 1]:
        p = line.split()
        prev.append(int(p[0]))
        nxt.append(int(p[1]))
    return prev, nxt


def read_xyz(f: int):
    lines = (REF / f"rt_is.{f}").read_text().strip().splitlines()
    n = int(lines[0])
    return np.array([[float(v) for v in l.split()[1:4]]
                     for l in lines[1: n + 1]])


def load(res: Path):
    prev, nxt, n = {}, {}, {}
    for f in range(FIRST, LAST + 1):
        p, x = read_pn(res, f)
        prev[f], nxt[f] = p, x
        n[f] = len(p)
    return prev, nxt, n


def chains(prev, nxt, nrows):
    out = []
    for f in range(FIRST, LAST + 1):
        for i in range(nrows[f]):
            if prev[f][i] < 0:
                ch = [(f, i)]
                cf, ci = f, i
                while cf <= LAST and ci < len(nxt[cf]) and nxt[cf][ci] >= 0:
                    ci = nxt[cf][ci]
                    cf += 1
                    ch.append((cf, ci))
                out.append(tuple(ch))
    return out


def pos(node, XYZ, nrows):
    f, i = node
    if i >= nrows[f]:
        return None
    return XYZ[f][i]


def acc(Xg, Xa, Xb):
    if Xg is None:
        return float("nan")
    return float(np.linalg.norm(Xb - 2 * Xa + Xg))


def show_case(title, ref_chain, tst_chain, XYZ, ref_n, rp, tp):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    print(f"  ref chain ({len(ref_chain)} nodes): "
          + " -> ".join(f"{f}:{i}" for f, i in ref_chain))
    if tst_chain is not None:
        print(f"  test chain ({len(tst_chain)} nodes): "
              + " -> ".join(f"{f}:{i}" for f, i in tst_chain))
    print(f"  {'frame':>8} {'ref node':>9} {'x,y,z':>24} "
          f"{'step':>6} {'acc':>6} | {'tst node':>9} {'step':>6} {'acc':>6}")
    # walk frames covered by either chain
    frames = sorted({f for f, _ in ref_chain} |
                    ({f for f, _ in tst_chain} if tst_chain else set()))
    rmap = dict(zip([f for f, _ in ref_chain], ref_chain))
    tmap = dict(zip([f for f, _ in tst_chain], tst_chain)) if tst_chain else {}
    rprev_node, tprev_node = None, None
    rgp, tgp = None, None
    for f in frames:
        rnode, tnode = rmap.get(f), tmap.get(f)
        Xr = pos(rnode, XYZ, ref_n) if rnode else None
        Xt = pos(tnode, XYZ, ref_n) if tnode else None
        rstep = (float(np.linalg.norm(Xr - pos(rprev_node, XYZ, ref_n)))
                 if Xr is not None and rprev_node is not None
                 and pos(rprev_node, XYZ, ref_n) is not None else float("nan"))
        tstep = (float(np.linalg.norm(Xt - pos(tprev_node, XYZ, ref_n)))
                 if Xt is not None and tprev_node is not None
                 and pos(tprev_node, XYZ, ref_n) is not None else float("nan"))
        racc = acc(pos(rgp, XYZ, ref_n) if rgp else None,
                   pos(rprev_node, XYZ, ref_n) if rprev_node else None, Xr) \
            if Xr is not None and rprev_node else float("nan")
        tacc = acc(pos(tgp, XYZ, ref_n) if tgp else None,
                   pos(tprev_node, XYZ, ref_n) if tprev_node else None, Xt) \
            if Xt is not None and tprev_node else float("nan")
        mark = "  " if rnode == tnode else ">>"
        print(f"{mark} {f:>8} {str(rnode):>9} "
              f"{('%.2f,%.2f,%.2f' % tuple(Xr)) if Xr is not None else '-':>24} "
              f"{rstep:6.3f} {racc:6.3f} | {str(tnode):>9} "
              f"{tstep:6.3f} {tacc:6.3f}")
        if rnode:
            rgp, rprev_node = rprev_node, rnode
        if tnode:
            tgp, tprev_node = tprev_node, tnode


def plot_case(title, ref_chain, tst_chain, XYZ, ref_n, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(title, fontsize=10)
    R = np.array([XYZ[f][i] for f, i in ref_chain
                  if i < ref_n[f]])
    ax1.plot(R[:, 0], R[:, 1], "o-", color="#2a6ebb", label="3dptv",
             markersize=7)
    ax2.plot(R[:, 0], R[:, 2], "o-", color="#2a6ebb", label="3dptv",
             markersize=7)
    if tst_chain:
        T = np.array([XYZ[f][i] for f, i in tst_chain
                      if i < ref_n[f]])
        ax1.plot(T[:, 0], T[:, 1], "s--", color="#c0392b",
                 label="openptv2", markersize=6)
        ax2.plot(T[:, 0], T[:, 2], "s--", color="#c0392b",
                 label="openptv2", markersize=6)
    # divergence node: first frame where they differ
    rm = dict(zip([f for f, _ in ref_chain], ref_chain))
    tm = dict(zip([f for f, _ in tst_chain], tst_chain)) if tst_chain else {}
    for f in sorted(set(rm) & set(tm)):
        if rm[f] != tm[f]:
            for ax, xy in ((ax1, (0, 1)), (ax2, (0, 2))):
                P = XYZ[f][rm[f][1]]
                ax.annotate("split", (P[xy[0]], P[xy[1]]),
                            textcoords="offset points", xytext=(8, 8),
                            fontsize=9, color="black",
                            arrowprops={"arrowstyle": "->"})
            break
    for ax, xl, yl in ((ax1, "x (mm)", "y (mm)"), (ax2, "x (mm)", "z (mm)")):
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=REPO / "scratch" / "_wp1_ab")
    ap.add_argument("--out", type=Path, default=REPO / "scratch" / "single_traj")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    tst_res = args.work.resolve() / "res"

    rp, rn, ref_n = load(REF)
    tp, tn, tst_n = load(tst_res)
    XYZ = {f: read_xyz(f) for f in range(FIRST, LAST + 1)}

    ref_chains = chains(rp, rn, ref_n)
    tst_chains = chains(tp, tn, tst_n)
    tst_set = set(tst_chains)
    node2tst = {}
    for c in tst_chains:
        for node in c:
            node2tst[node] = c
    node2ref = {}
    for c in ref_chains:
        for node in c:
            node2ref[node] = c
    ref_edges = set()
    for c in ref_chains:
        ref_edges |= {(c[k], c[k + 1]) for k in range(len(c) - 1)}
    tst_edges = set()
    for c in tst_chains:
        for k in range(len(c) - 1):
            a, b = c[k], c[k + 1]
            if a[1] < ref_n[a[0]] and b[1] < ref_n[b[0]]:
                tst_edges.add((a, b))

    cases = []
    # 1-2. longest mid-splits
    mids = []
    for c in ref_chains:
        E = [(c[k], c[k + 1]) for k in range(len(c) - 1)]
        if len(E) < 3:
            continue
        hit = [e in tst_edges for e in E]
        if any(hit) and not all(hit):
            miss = [k for k, h in enumerate(hit) if not h]
            if any(0 < k < len(E) - 1 for k in miss):
                mids.append(c)
    mids.sort(key=len, reverse=True)
    for k, c in enumerate(mids[:2]):
        # test chain through the split node
        E = [(c[i], c[i + 1]) for i in range(len(c) - 1)]
        hit = [e in tst_edges for e in E]
        split_at = next(i for i, h in enumerate(hit) if not h)
        tchain = node2tst.get(c[split_at], None)
        cases.append((f"case{k + 1}-midsplit len{len(c)}", c, tchain))

    # 3. a merge: test edge joining two ref chains
    merged = None
    for c in ref_chains:
        E = [(c[i], c[i + 1]) for i in range(len(c) - 1)]
        for a, b in E:
            if (a, b) not in tst_edges and a[1] < len(tn[a[0]]):
                cc = tn[a[0]][a[1]]
                if cc >= 0:
                    b2 = (a[0] + 1, cc)
                    if b2 in node2ref and node2ref[b2] != c:
                        merged = (c, node2tst.get(a))
                        break
        if merged:
            break
    if merged:
        cases.append((f"case3-merge len{len(merged[0])}", merged[0], merged[1]))

    # 4. longest fully-lost multi-node chain
    lost = [c for c in ref_chains if len(c) > 2 and not any(
        (c[i], c[i + 1]) in tst_edges for i in range(len(c) - 1))]
    lost.sort(key=len, reverse=True)
    if lost:
        c = lost[0]
        cases.append((f"case4-lost len{len(c)}", c, node2tst.get(c[0])))

    # 5. ghost extension: ref singleton that test links, longest test chain
    ghost = []
    for c in ref_chains:
        if len(c) == 1:
            (f, i) = c[0]
            if i < len(tn[f]) and tn[f][i] >= 0:
                ghost.append((c, node2tst.get((f, i))))
    ghost.sort(key=lambda t: len(t[1] or ()), reverse=True)
    if ghost:
        c, t = ghost[0]
        cases.append(("case5-ghost", c, t))

    print(f"selected {len(cases)} cases")
    for title, c, t in cases:
        show_case(title, c, t, XYZ, ref_n, rp, tp)
        plot_case(title, c, t, XYZ, ref_n, out / (title + ".png"))

    args_t = ", ".join(title for title, _, _ in cases)
    print(f"\ncases: {args_t}")


if __name__ == "__main__":
    main()
