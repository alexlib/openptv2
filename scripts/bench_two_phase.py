"""Can the two-phase tracker solve the hard crossings? Plus a velocity-aware fix.

Two-phase (plugins/two_phase_tracking.py): phase 1 = 3D KD-tree candidates
within v_max; phase 2 = Hungarian assignment on 2D pixel-distance costs.
Memoryless: no velocity prediction, no history.

Prediction tested here:
  (a) two-phase CANNOT do steady crossings (bounce bias: at an X-crossing
      the swap assignment always has lower total cost than the cross), and
  (b) whether adding constant-velocity prediction to its phase-1 search
      points (pred = x + v) fixes crossings AND maneuvers.

Engines: corr-default (reference), two-phase raw, two-phase+velocity.
Same S-scenes/seeds/scoring as synth_crossing.

Run from repo root:
    uv run python -u scripts/bench_two_phase.py [--reps N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from synth_crossing import (  # noqa: E402
    REPO,
    load_optics,
    make_truth,
    project,
    setup_work,
    write_scene,
)

V_MAX = 3.3  # mirrors the dv +/-1.9 box (1.9*sqrt(3))


def load_rt_positions(work: Path, nf: int, first: int):
    frames = []
    for f in range(nf):
        lines = (work / "res" / f"rt_is.{first + f}").read_text().splitlines()
        n = int(lines[0])
        frames.append(np.array(
            [[float(v) for v in l.split()[1:4]] for l in lines[1: n + 1]]))
    return frames


def project_points(P, cpar, cals):
    from synth_crossing import NCAMS

    n = len(P)
    xy = np.full((n, NCAMS * 2), np.nan)
    for i, (x, y, z) in enumerate(P):
        for ci, (px, py) in enumerate(project(cpar, cals, x, y, z)):
            xy[i, 2 * ci] = px
            xy[i, 2 * ci + 1] = py
    return np.nan_to_num(xy)


def make_leaves(positions, cpar, cals):
    return [project_points(P, cpar, cals) for P in positions]


def run_productized(work, cpar, cals):
    """The productized plugin class (velocity + gaps + re-projection)."""
    from synth_crossing import FIRST as _F
    from synth_crossing import NF as _N

    from openptv2.plugins.two_phase_tracking import (
        TwoPhaseTracker,
        TwoPhaseTrackerConfig,
    )

    positions = load_rt_positions(work, _N, _F)
    leaves = make_leaves(positions, cpar, cals)
    cfg = TwoPhaseTrackerConfig(v_max=V_MAX, leaf_weight=1.0,
                                use_velocity=True, cost_mode="projected",
                                max_gap=2)
    tr = TwoPhaseTracker(cfg)
    return tr.track_frames(
        [np.asarray(p) for p in positions], leaves,
        project_fn=lambda P: project_points(np.asarray(P), cpar, cals))


def run_two_phase(work, cpar, cals, nf, first, use_velocity: bool):
    from openptv2.plugins.two_phase_tracking import _match_two_phase_frame

    positions = load_rt_positions(work, nf, first)
    leaves = make_leaves(positions, cpar, cals)
    n = len(positions[0])
    vel = {i: np.zeros(3) for i in range(n)}
    links = []  # (t0, pid0, t1, pid1)
    for t in range(nf - 1):
        pts0, pts1 = positions[t], positions[t + 1]
        if use_velocity:
            pred = np.array([pts0[i] + vel[i] for i in range(n)])
            # costs must be evaluated AT the predicted positions:
            # re-project predictions to pixels for the leaf signature
            pred_leaves = project_points(pred, cpar, cals)
        else:
            pred, pred_leaves = pts0, leaves[t]
        got = _match_two_phase_frame(
            pred, pts1, pred_leaves, leaves[t + 1],
            np.arange(n, dtype=np.int32), np.arange(n, dtype=np.int32),
            V_MAX, 1.0)
        for pid0, pid1 in got:
            links.append((t, pid0, t + 1, pid1))
        if use_velocity:
            new_vel = {}
            for pid0, pid1 in got:
                new_vel[pid1] = pts1[pid1] - pts0[pid0]
            vel = {i: new_vel.get(i, np.zeros(3)) for i in range(n)}
    return links


def score_links(links, rows_pf, nf, first):
    # expected row->row links per pid from per-frame row maps (gap-aware:
    # a pid absent at f-1 links from its last-seen row).
    expected = set()
    for pid_frames in _pid_frames(rows_pf, nf, first):
        for (t0, r0), (t1, r1) in zip(pid_frames, pid_frames[1:]):
            expected.add((t0, r0, t1, r1))
    got = set(links)
    n_ok = len(expected & got)
    n_tot = len(expected)
    cross_ok = all(e in got for e in expected if e[2] == 4)
    return (n_ok / n_tot if n_tot else 1.0), cross_ok


def _pid_frames(rows_pf, nf, first):
    by_pid = {}
    for f in range(nf):
        for pid, row in rows_pf[first + f].items():
            by_pid.setdefault(pid, []).append((f, row))
    return list(by_pid.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=8)
    args = ap.parse_args()

    from synth_crossing import FIRST, NF

    cpar, cals = load_optics()
    scenarios = [
        ("S1 head-on", {"v": 0.5}),
        ("S8 kick", {"v": 0.5, "maneuver": "kick"}),
        ("S11 kick+noise", {"v": 0.5, "maneuver": "kick", "noise": 0.07}),
        ("S4 head-on+noise", {"v": 0.5, "noise": 0.07}),
        ("S13 gap-occl", {"v": 0.5, "dropout": True}),
    ]
    print(f"{'scenario':16}{'engine':16}{'recall':>9}{'cross-ok':>10}",
          flush=True)
    for sname, kw in scenarios:
        noisy = kw.get("noise", 0.0) > 0
        reps = args.reps if noisy else 3
        # prototype harnesses assume constant-N frames; the dropout scene
        # (varying N) runs productized-only.
        engines = ("productized",) if kw.get("dropout") else (
            "two-phase", "two-phase+vel", "productized")
        for ename in engines:
            recs, crs = [], []
            for rep in range(reps):
                work = REPO / "scratch" / "_synth_2p"
                setup_work(work)
                truth = make_truth(seed=1000 + rep, **kw)
                rows_pf = write_scene(work, truth, cpar, cals)
                if ename == "productized":
                    links = run_productized(work, cpar, cals)
                else:
                    links = run_two_phase(work, cpar, cals, NF, FIRST,
                                          use_velocity=ename.endswith("+vel"))
                r, c = score_links(links, rows_pf, NF, FIRST)
                recs.append(r)
                crs.append(c)
            print(f"{sname:16}{ename:16}{np.mean(recs):>8.1%}  "
                  f"{sum(crs)}/{len(crs)}", flush=True)


if __name__ == "__main__":
    main()
