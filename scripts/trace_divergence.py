"""Find where test history first diverges from ref along hard-link sources.

Many union-miss links are downstream victims: at decision time the test
prediction was already 20-36 mm off because an UPSTREAM link diverged.
This script walks both chains back from each hard link's source and names
the first divergence frame + the competing choice -- the actual decision
to adjudicate by manual vote.
"""

from collections import Counter
from pathlib import Path

DS = Path(r"C:\Users\alex\Downloads\HiDImaging\wp1_10_images")
REF = DS / "res_ground_truth_backup"
TST = Path("scratch/_ens_lr1/res")
FIRST, LAST = 100001, 100010


def load(d):
    prev, nxt = {}, {}
    for f in range(FIRST, LAST + 1):
        l = (d / f"ptv_is.{f}").read_text().splitlines()
        n = int(l[0])
        p, x = [], []
        for line in l[1: n + 1]:
            s = line.split()
            p.append(int(s[0]))
            x.append(int(s[1]))
        prev[f], nxt[f] = p, x
    return prev, nxt


def chain_back(prev, f, i):
    out = [(f, i)]
    while out[-1][0] > FIRST:
        ff, ii = out[-1]
        p = prev[ff][ii] if ii < len(prev[ff]) else -1
        if p < 0:
            break
        out.append((ff - 1, p))
    return out


def main():
    rp, rn = load(REF)
    tp, tn = load(TST)
    import csv
    hard = list(csv.DictReader(open("scratch/hard_links/hard_links.csv")))
    div_at = Counter()
    print(f"{'frame':>8}{'row':>6}  divergence (frame, ref-node -> test-node)")
    shown = 0
    for r in hard:
        f, i, p = int(r["frame"]), int(r["row"]), int(r["ref_prev"])
        rc = chain_back(rp, f - 1, p)  # rc[0] == tc[0] == (f-1, p)
        tc = chain_back(tp, f - 1, p)
        div = None
        for k in range(1, max(len(rc), len(tc))):
            rk = rc[k] if k < len(rc) else None
            tk = tc[k] if k < len(tc) else None
            if rk != tk:
                div = (rc[k - 1], rk, tk)
                break
        if div is None:
            print(f"{f:>8}{i:>6}  chains identical "
                  f"(len {len(rc)}); miss is LOCAL to this link")
        else:
            prev_node, rnode, tnode = div
            print(f"{f:>8}{i:>6}  after {prev_node}: ref->{rnode} "
                  f"test->{tnode}")
            div_at[(prev_node, rnode, tnode)] += 1
    print(f"\n{len(div_at)} unique upstream divergences:")
    for k, v in div_at.most_common():
        print(f"  {v}x  after {k[0]}: ref->{k[1]} test->{k[2]}")


if __name__ == "__main__":
    main()
