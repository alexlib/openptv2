"""Trace WHICH ground-truth tracks break under a tracker config and why.

Instead of scoring only aggregate metrics, walk each GT track and report:
  * coverage C   - fraction of frames the tracker kept
  * n_frag       - how many predicted fragments map to this true track
  * needed_win   - the smallest uniform search window (mm) that would have
    covered every observed step of this true track (kinematics demand)
  * break frames - list of (frame, gap, |step|max) for each disconnect

Usage:
  uv run python scripts/diag_track_breaks.py [--tracker priority_segment_3d]
        [--top N]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

logging.getLogger("openptv2").setLevel(logging.CRITICAL)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_utils as bu  # noqa: E402

EPS = 1.0


def frames_dict(tracks):
    out = {}
    for tid, pts in tracks.items():
        for f, x, y, z in pts:
            out.setdefault(f, {})[tid] = np.array([x, y, z])
    return out


def _match(tf, pf, all_frames):
    """Same nearest-neighbour logic as openptv2 metrics, kept per-frame."""
    match = {}
    for f in all_frames:
        if f not in tf or f not in pf:
            match[f] = {}
            continue
        ti = list(tf[f].keys())
        pi = list(pf[f].keys())
        tree = cKDTree(np.array([tf[f][t] for t in ti]))
        d, idx = tree.query(np.array([pf[f][p] for p in pi]))
        match[f] = {pid: ti[ix] for pid, ix, dd in zip(pi, idx, d) if dd <= EPS}
    return match


def analyze_track(gtid, pts, match):
    cov = set()
    frag = set()
    for f, _, _, _ in pts:
        for pid, tidv in match.get(f, {}).items():
            if tidv == gtid:
                cov.add(f)
                frag.add(pid)

    # kinematics: what uniform window would have kept this track contiguous
    need = 0.0
    breaks = []
    sorted_pts = [np.array(p) for p in sorted(pts)]
    prev = sorted_pts[0]
    for cur in sorted_pts[1:]:
        step = np.abs(np.asarray(cur[1:], float) - np.asarray(prev[1:], float))
        need = max(need, float(step.max()))
        if float(step.max()) > bu.BASE_OVERRIDES["dvxmax"]:
            breaks.append((int(cur[0]), int(cur[0] - prev[0]), round(float(step.max()), 2)))
        prev = cur

    C = len(cov) / max(1, len(pts))
    return {"gtid": gtid, "cov": C, "n_frag": len(frag),
            "need": round(need, 2), "breaks": breaks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", default="priority_segment_3d")
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    tt = bu.build_true_tracks(bu.read_gt_frames())
    pred0, _ = bu.run_single_tracker(args.tracker, bu.BASE_OVERRIDES)
    pf = frames_dict(pred0)
    tf = frames_dict(tt)
    allf = sorted(set(tf) | set(pf))
    match = _match(tf, pf, allf)

    rows = [analyze_track(gid, pts, match) for gid, pts in sorted(tt.items())]
    rows.sort(key=lambda r: (-r["n_frag"], -r["cov"]))

    print(f"tracker={args.tracker} eps={EPS}")
    print(f"{'gt':>5} {'C':>5} {'frag':>4} {'win':>7} breaks(frame,gap,max|step|)")
    for r in rows[: args.top]:
        br = ", ".join(f"({a},{b},{c})" for a, b, c in r["breaks"]) or "-"
        print(f"{r['gtid']:>5} {r['cov']:>5.2f} {r['n_frag']:>4} "
              f"{r['need']:>7.2f} {br}")

    clean = [r for r in rows if r["n_frag"] == 1 and r["cov"] == 1.0]
    print(f"\n{len(clean)}/{len(rows)} tracks kept clean at dvx={bu.BASE_OVERRIDES['dvxmax']}")
    newsup = sorted({r['need'] for r in rows if r['n_frag'] > 1})
    print(f"fragmented tracks would need window > {bu.BASE_OVERRIDES['dvxmax']}: {newsup}")


if __name__ == "__main__":
    main()