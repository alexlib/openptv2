"""Residual autopsy at matched parameters (dacc=1.9): WHAT differs, not how much.

Compares 3dptv reference chains vs openptv2 chains on the same rt_is.*:
  1. chain census (lengths, singletons, full-span, ghost/added chains)
  2. break classification (head / tail / mid-split / merge / lost)
  3. smoothness (3-point acc distributions along each engine's own chains)
  4. contest analysis (at every divergence, whose link is smoother under
     whose history -- is openptv2 wrong, or alternatively-valid?)

Run from repo root (after a dacc=1.9 run in --work):
    uv run python -u scripts/analyze_residual.py [--work scratch/_wp1_ab]
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


def chains(prev, nxt, nrows, first, last, row_cap=None):
    """Maximal chains from heads (prev<0). Returns list of node tuples."""
    out = []
    for f in range(first, last + 1):
        cap = nrows[f] if row_cap is None else min(nrows[f], row_cap[f])
        for i in range(min(cap, len(prev[f]))):
            if prev[f][i] < 0:
                ch = [(f, i)]
                cf, ci = f, i
                while (cf <= last and ci < len(nxt[cf])
                       and nxt[cf][ci] >= 0):
                    ci = nxt[cf][ci]
                    cf += 1
                    ch.append((cf, ci))
                out.append(tuple(ch))
    return out


def edges_of(ch):
    return {(ch[k], ch[k + 1]) for k in range(len(ch) - 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=REPO / "scratch" / "_wp1_ab")
    args = ap.parse_args()
    work = args.work.resolve()
    tst_res = work / "res"

    rp, rn, ref_n = load(REF)
    tp, tn, tst_n = load(tst_res)
    print(f"rows/frame ref : {[ref_n[f] for f in range(FIRST, LAST + 1)]}")
    print(f"rows/frame test: {[tst_n[f] for f in range(FIRST, LAST + 1)]}")

    XYZ = {f: read_xyz(f) for f in range(FIRST, LAST + 1)}

    ref_chains = chains(rp, rn, ref_n, FIRST, LAST)
    # test chains restricted to shared rows for head-finding, but allowed to
    # wander into added rows when followed
    tst_chains = chains(tp, tn, tst_n, FIRST, LAST)
    tst_shared = chains(tp, tn, tst_n, FIRST, LAST,
                        row_cap={f: ref_n[f] for f in range(FIRST, LAST + 1)})

    # ---- 1. census ----
    print("\n== chain census ==")
    for tag, chs in (("ref ", ref_chains), ("test", tst_chains)):
        L = np.array([len(c) for c in chs])
        import collections
        hist = collections.Counter(len(c) for c in chs)
        print(f"{tag}: n={len(chs)} mean_len={L.mean():.2f} "
              f"singletons={hist[1]} fullspan={hist[LAST - FIRST + 1]}")
        print(f"   len histogram: {dict(sorted(hist.items()))}")

    ghost_chains = [c for c in tst_chains
                    if any(i >= ref_n[f] for f, i in c)]
    print(f"test chains touching added rows: {len(ghost_chains)} "
          f"({len(ghost_chains) / len(tst_chains):.1%} of test chains)")

    # ---- 2. break classification ----
    ref_edge_set = set()
    for c in ref_chains:
        ref_edge_set |= edges_of(c)
    tst_edge_set = set()
    for c in tst_chains:
        # only shared-row edges are comparable
        for a, b in edges_of(c):
            if a[1] < ref_n[a[0]] and b[1] < ref_n[b[0]]:
                tst_edge_set.add((a, b))

    # node -> ref chain id
    node2ref = {}
    for cid, c in enumerate(ref_chains):
        for node in c:
            node2ref[node] = cid

    exact = head = tail = mid = lost = 0
    merges = 0
    merge_examples = []
    for c in ref_chains:
        E = [(c[k], c[k + 1]) for k in range(len(c) - 1)]
        if not E:
            continue
        hit = [e in tst_edge_set for e in E]
        if all(hit):
            exact += 1
            continue
        if not any(hit):
            lost += 1
            continue
        # partial: where are the misses?
        miss_idx = [k for k, h in enumerate(hit) if not h]
        if all(k == 0 for k in miss_idx):
            head += 1
        elif all(k == len(E) - 1 for k in miss_idx):
            tail += 1
        else:
            mid += 1
        # merge: test edge leaving a node of this chain into another chain?
        for k in miss_idx:
            a = E[k][0]
            if a[0] <= LAST and a[1] < len(tn[a[0]]) and tn[a[0]][a[1]] >= 0:
                b2 = (a[0] + 1, tn[a[0]][a[1]])
                if (b2[1] < ref_n[b2[0]] and node2ref.get(b2, -1) != cid
                        and b2 in node2ref):
                    merges += 1
                    if len(merge_examples) < 5:
                        merge_examples.append(
                            (a, E[k][1], b2, node2ref[b2]))
                    break
    n_multi = len([c for c in ref_chains if len(c) > 1])
    print(f"\n== ref multi-node chains: {n_multi} ==")
    print(f"exact={exact} head-miss={head} tail-miss={tail} mid-split={mid} "
          f"lost={lost}")
    print(f"chains where test merged away mid-chain: {merges}")
    for a, bref, btest, ocid in merge_examples:
        print(f"   ref {a}->{bref}, test {a}->{btest} "
              f"(joins ref chain {ocid})")

    # test-only edges (neither endpoint pair in ref): ghosts + merges
    ref_nodes = set(node2ref)
    test_only = [e for e in tst_edge_set if e not in ref_edge_set]
    print(f"\ntest-only shared-row edges: {len(test_only)} "
          f"({len(test_only) / max(len(tst_edge_set), 1):.1%} of test edges)")

    # ---- 3. smoothness: acc along each engine's OWN chains ----
    def acc_list(chs):
        out = []
        for c in chs:
            for k in range(1, len(c) - 1):
                (f0, i0), (f1, i1), (f2, i2) = c[k - 1], c[k], c[k + 1]
                if (f1 == f0 + 1 and f2 == f1 + 1 and i0 < ref_n[f0]
                        and i1 < ref_n[f1] and i2 < ref_n[f2]):
                    X0, X1, X2 = XYZ[f0][i0], XYZ[f1][i1], XYZ[f2][i2]
                    out.append(float(np.linalg.norm(X2 - 2 * X1 + X0)))
        return np.array(out)

    ra, ta = acc_list(ref_chains), acc_list(tst_chains)
    print("\n== smoothness (3-point acc on own chains, shared rows) ==")
    for tag, a in (("ref ", ra), ("test", ta)):
        print(f"{tag}: n={len(a)} mean={a.mean():.4f} "
              f"p50={np.median(a):.4f} p90={np.percentile(a, 90):.4f} "
              f"p99={np.percentile(a, 99):.4f} max={a.max():.4f}")

    # ---- 4. contest analysis at divergences ----
    # ref edge (a->b) missing, test has (a->c): compare acc of each choice
    # under REF history (grandparent from ref chain) and under TEST history.
    def gp_of(node, prev_map):
        f, i = node
        if f - 1 < FIRST or i >= len(prev_map[f]):
            return None
        p = prev_map[f][i]
        if p < 0 or p >= ref_n[f - 1]:
            return None
        return (f - 1, p)

    smoother_test = smoother_ref = ties = 0
    inadmissible = 0  # test choice violates dv/dacc under ref history
    acc_pairs = []
    for c in ref_chains:
        for k in range(len(c) - 1):
            a, b = c[k], c[k + 1]
            if (a, b) in tst_edge_set:
                continue
            if a[1] >= len(tn[a[0]]):
                continue
            cc = tn[a[0]][a[1]]
            if cc < 0 or cc >= ref_n[a[0] + 1]:
                continue
            cnode = (a[0] + 1, cc)
            g = gp_of(a, rp)
            if g is None:
                continue
            Xa, Xb, Xc, Xg = XYZ[a[0]][a[1]], XYZ[b[0]][b[1]], \
                XYZ[cnode[0]][cnode[1]], XYZ[g[0]][g[1]]
            acc_ref = float(np.linalg.norm(Xb - 2 * Xa + Xg))
            acc_tst = float(np.linalg.norm(Xc - 2 * Xa + Xg))
            acc_pairs.append((acc_ref, acc_tst))
            if acc_tst < acc_ref - 1e-9:
                smoother_test += 1
            elif acc_ref < acc_tst - 1e-9:
                smoother_ref += 1
            else:
                ties += 1
            if acc_tst >= 1.9:
                inadmissible += 1
    P = np.array(acc_pairs)
    print(f"\n== contests (ref edge missed, test chose elsewhere; n={len(P)}) ==")
    print(f"test choice smoother: {smoother_test} "
          f"({smoother_test / max(len(P), 1):.1%})")
    print(f"ref choice smoother : {smoother_ref} "
          f"({smoother_ref / max(len(P), 1):.1%})")
    print(f"ties: {ties}; test choice acc>=dacc(1.9): {inadmissible}")
    if len(P):
        print(f"mean acc: ref-choice {P[:, 0].mean():.4f} vs "
              f"test-choice {P[:, 1].mean():.4f}")


if __name__ == "__main__":
    main()
