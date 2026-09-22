"""Legacy track.c vs openptv2 link-resolution, side by side.

Both trackers score candidate links identically (verified: the `rr` formula and
the acc/angle gate in track_kernels_corr.py match track.c line-for-line). They
diverge at the LAST step -- deciding who keeps a contested candidate.

track.c (lines 598-646): one pass. A particle whose best candidate is already
taken either evicts the holder (if its finaldecis is better) or is set to
next = -2 and is DONE. The evicted holder is also set to -2 and is DONE. Neither
ever falls back to its second choice -- the author left the commented-out
printfs ("zweite Wahl fuer %d waere") at lines 617-623 and 635-641 showing the
second choice was inspected but deliberately not used.

openptv2 (track_kernels_corr.py:1273-1282) adds a third phase that track.c has
no counterpart for: every loser walks its remaining candidates and claims the
first unclaimed one.

This script runs both resolvers over the same decision table and reports the
links that exist in one and not the other.

Run from repo root:
    uv run python scripts/diag_conflict_parity.py
"""

from __future__ import annotations

NONE = -2
FREE = -1


def resolve_legacy(inlist, decis, linkdecis, n_next):
    """track.c lines 588-646. Losers die; no fallback."""
    nxt = [NONE] * len(inlist)
    finaldecis = [1e6] * len(inlist)
    prev = [FREE] * n_next

    for h in range(len(inlist)):
        if inlist[h] > 0:
            order = sorted(range(inlist[h]), key=lambda i: decis[h][i])
            finaldecis[h] = decis[h][order[0]]
            nxt[h] = linkdecis[h][order[0]]

    for h in range(len(inlist)):
        if inlist[h] <= 0:
            continue
        cand = nxt[h]
        if prev[cand] == FREE:
            prev[cand] = h
        elif finaldecis[prev[cand]] > finaldecis[h]:
            nxt[prev[cand]] = NONE  # evicted holder is done, no retry
            prev[cand] = h
        else:
            nxt[h] = NONE  # loser is done, no retry
    return nxt


def resolve_openptv2(inlist, decis, linkdecis, n_next):
    """track_kernels_corr.py 1226-1286. Adds phase 3: losers retry."""
    nxt = [NONE] * len(inlist)
    finaldecis = [1e6] * len(inlist)
    prev = [FREE] * n_next
    order_of = {}

    for h in range(len(inlist)):
        if inlist[h] > 0:
            order = sorted(range(inlist[h]), key=lambda i: decis[h][i])
            order_of[h] = order
            finaldecis[h] = decis[h][order[0]]
            nxt[h] = linkdecis[h][order[0]]

    for h in range(len(inlist)):
        if inlist[h] <= 0:
            continue
        cand = nxt[h]
        if prev[cand] == FREE:
            prev[cand] = h
        elif finaldecis[prev[cand]] > finaldecis[h]:
            nxt[prev[cand]] = NONE
            prev[cand] = h
        else:
            nxt[h] = NONE

    # --- phase 3: the part track.c does not have ---
    for h in range(len(inlist)):
        if inlist[h] > 1 and nxt[h] == NONE:
            for ti in order_of[h][1:]:
                cand = linkdecis[h][ti]
                if prev[cand] == FREE:
                    nxt[h] = cand
                    finaldecis[h] = decis[h][ti]
                    prev[cand] = h
                    break
    return nxt


def report(name, inlist, decis, linkdecis, n_next, speeds=None):
    a = resolve_legacy(inlist, decis, linkdecis, n_next)
    b = resolve_openptv2(inlist, decis, linkdecis, n_next)
    print(f"\n=== {name} ===")
    print(f"{'particle':>9}  {'track.c':>12}  {'openptv2':>12}   note")
    for h in range(len(inlist)):
        note = ""
        if a[h] != b[h]:
            note = "<-- DIVERGES"
            if speeds and b[h] in speeds:
                note += f"  (openptv2 links a {speeds[b[h]]} step)"
        la = "dropped" if a[h] == NONE else f"-> {a[h]}"
        lb = "dropped" if b[h] == NONE else f"-> {b[h]}"
        print(f"{h:>9}  {la:>12}  {lb:>12}   {note}")
    print(f"  links: track.c={sum(x != NONE for x in a)}  "
          f"openptv2={sum(x != NONE for x in b)}")
    return a, b


def demo():
    # Two frame-1 particles both rank frame-2 candidate 0 as their best.
    # Particle 1 loses the contest. Its second choice is candidate 1, which is
    # a longer (faster) step -- rr penalises displacement via dl/lmax, so a
    # particle's later choices are generally its longer ones.
    inlist = [2, 2]
    decis = [[0.10, 0.80], [0.40, 0.90]]
    linkdecis = [[0, 1], [0, 1]]
    speeds = {1: "longer"}
    a, b = report("contested candidate", inlist, decis, linkdecis, 2, speeds)
    assert a == [0, NONE], a
    assert b == [0, 1], b

    # Three-way pile-up on one candidate: track.c drops two, openptv2 salvages
    # whatever is still free.
    inlist3 = [2, 2, 2]
    decis3 = [[0.10, 0.50], [0.20, 0.60], [0.30, 0.70]]
    linkdecis3 = [[0, 1], [0, 1], [0, 2]]
    a3, b3 = report("three-way pile-up", inlist3, decis3, linkdecis3, 3)
    assert a3 == [0, NONE, NONE], a3
    assert b3 == [0, 1, 2], b3

    # No contest -> identical. Guards against the resolvers differing anywhere
    # other than the contested path.
    a4, b4 = report("no contest", [1, 1], [[0.1], [0.2]], [[0], [1]], 2)
    assert a4 == b4 == [0, 1], (a4, b4)

    print("\nself-check OK: the two resolvers agree unless a candidate is contested.")


if __name__ == "__main__":
    demo()
