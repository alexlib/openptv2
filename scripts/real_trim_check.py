"""Effect of trimming doubtful trajectory ends on REAL data (answer-free scores).

    uv run python scripts/real_trim_check.py RUN_DIR LAST THR [THR ...]

RUN_DIR holds parameters_*.yaml, cal/ and res/run.zarr with linkage 'ptv_is'
(e.g. a scratch copy made by real_retrack.py). Prints realism_metrics scores before
and after trimming up to 3 end points with ghost probability above each THR.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import realism_metrics as rm  # noqa: E402
from synth_bench import _experiment  # noqa: E402

from openptv2.point_quality import (  # noqa: E402
    fit_scale,
    frame_rcm,
    ghost_probability,
)
from openptv2.storage import RunStore  # noqa: E402


def chains_with_ghost(run: Path, last: int):
    exp = _experiment(run, next(run.glob("parameters_*.yaml")), 1, last)
    s = RunStore.open(run, mode="r")
    pos, nxt, prv, nc, rcm = {}, {}, {}, {}, {}
    for f in range(1, last + 1):
        p, n, q = s.read_linkage(f, "ptv_is")
        _, cam = s.read_correspondences(f)
        pos[f], nxt[f], prv[f], nc[f] = q[:, :3], n, p, (cam >= 0).sum(1)
        rcm[f] = frame_rcm(s, f, exp.cals, exp.cpar)
    scale = fit_scale(
        np.vstack([pos[f] for f in pos]),
        np.concatenate([rcm[f] for f in pos]),
        np.concatenate([nc[f] for f in pos]),
    )
    pg = {f: ghost_probability(pos[f], rcm[f], nc[f], scale) for f in pos}
    out = []
    for f in range(1, last + 1):
        for i in range(len(pos[f])):
            if prv[f][i] >= 0 and f > 1:
                continue
            c, k, g, ff, ii = [], [], [], f, i
            while True:
                c.append(pos[ff][ii])
                k.append(nc[ff][ii])
                g.append(pg[ff][ii])
                j = nxt[ff][ii]
                if j < 0 or ff == last:
                    break
                ff, ii = ff + 1, int(j)
            out.append((np.array(c), np.array(k), np.array(g)))
    return out


def trim(ch, thr, max_trim=3):
    res = []
    for c, k, g in ch:
        a, b = 0, len(c)
        while a < min(max_trim, len(c)) and g[a] > thr:
            a += 1
        while b > a and len(c) - b < max_trim and g[b - 1] > thr:
            b -= 1
        if b - a >= 1:
            res.append((c[a:b], k[a:b], g[a:b]))
    return res


def show(label, ch, n_frames, n_points0):
    m = rm.metrics([(c, k) for c, k, _ in ch], n_frames)
    pts = sum(len(c) for c, _, _ in ch)
    print(
        f"{label:>10}: points {pts / n_points0:.3f}  trajectories {m['trajectories']:6d}  "
        f"mean_len {m['mean_len']:5.2f}  jump_share {m['jump_share']:.4f}  "
        f"jitter_z_p90 {m['jitter_z_p90']:.4f}  never4 {m.get('never4_share', float('nan')):.4f}"
    )


if __name__ == "__main__":
    run, last = Path(sys.argv[1]), int(sys.argv[2])
    ch = chains_with_ghost(run, last)
    n0 = sum(len(c) for c, _, _ in ch)
    show("no trim", ch, last, n0)
    for t in sys.argv[3:]:
        show(f"trim>{t}", trim(ch, float(t)), last, n0)
