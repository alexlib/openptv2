"""Tune app_weight on S15 (appearance-decides kick) + no-regression spot.

S15: bright A (2000) vs dim B (500), kick maneuver + noise + 10% sumg
jitter. Kinematics favor the swap; appearance favors truth. Expect a
fail -> pass transition as app_weight grows, with S1/S8/S11 unchanged.

Run from repo root:
    uv run python -u scripts/sweep_app_weight.py [--reps N]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from synth_crossing import (  # noqa: E402
    REPO,
    load_optics,
    make_truth,
    run_tracker,
    score,
    setup_work,
    write_scene,
)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=8)
    args = ap.parse_args()

    cpar, cals = load_optics()
    s15 = {"v": 0.5, "maneuver": "kick", "noise": 0.07,
           "_sumg": {0: 2000.0, 1: 500.0}, "_sj": 0.10}
    sfn = lambda pid, frame: {0: 2000.0, 1: 500.0}.get(pid, 1000.0)  # noqa: E731

    print(f"{'w_app':>7}{'S15 recall':>12}{'cross-ok':>10}  "
          f"{'S1':>6}{'S8':>6}{'S11r':>7}", flush=True)
    for w in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8):
        recs, crs = [], []
        for rep in range(args.reps):
            work = REPO / "scratch" / "_synth_app"
            setup_work(work)
            truth = make_truth(seed=1000 + rep, **{k: v for k, v in
                                                   s15.items()
                                                   if not k.startswith("_")})
            rows_pf = write_scene(work, truth, cpar, cals, sumg_fn=sfn,
                                  sumg_jitter=0.10, seed=2000 + rep)
            run_tracker(work, 0, 0, app_weight=w)
            r, c = score(work, rows_pf)
            recs.append(r)
            crs.append(c)
        # no-regression spot checks (S1/S8 single, S11 mean recall)
        spots = []
        for skw in ({"v": 0.5},
                    {"v": 0.5, "maneuver": "kick"},
                    {"v": 0.5, "maneuver": "kick", "noise": 0.07}):
            work = REPO / "scratch" / "_synth_app"
            setup_work(work)
            truth = make_truth(seed=5, **skw)
            rows_pf = write_scene(work, truth, cpar, cals)
            run_tracker(work, 0, 0, app_weight=w)
            r, _ = score(work, rows_pf)
            spots.append(r)
        print(f"{w:>7.2f}{np.mean(recs):>11.1%}  {sum(crs)}/{len(crs)}  "
              f"{spots[0]:>5.0%}{spots[1]:>6.0%}{spots[2]:>7.1%}", flush=True)


if __name__ == "__main__":
    main()
