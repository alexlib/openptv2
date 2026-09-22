"""Prototype: post-tracking trajectory triage with TypeSafe (Jev) Noul judgments.

Reads trajectories from a run.zarr, summarises each one in kinematic terms, and
asks Jev "does this look like a spurious link?". Tracking itself is untouched.

    uv run --with typesafe-sdk python scripts/track_triage_typesafe.py RUN.zarr --dry-run
    TYPESAFE_API_KEY=... uv run --with typesafe-sdk python scripts/track_triage_typesafe.py RUN.zarr

--dry-run prints the state/question for a few trajectories, no API call.
Output: per-trajectory p_spurious plus a plain-code baseline for comparison.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from openptv2.storage.zarr_store import read_zarr_trajectories

QUESTION = (
    "Is this trajectory likely a spurious link (an ID swap between two particles, "
    "or a ghost) rather than one real particle moving through a smooth flow?"
)
CRITERIA = (
    "Real tracks change speed and direction gradually between frames. Suspect: a "
    "step much larger than its neighbours, a sharp reversal or turn, or large "
    "acceleration relative to speed. Very short tracks give weak evidence."
)


def summarise(traj) -> dict:
    pos = np.asarray(traj.pos()) * 1000.0  # m -> mm
    step = np.diff(pos, axis=0)
    speed = np.linalg.norm(step, axis=1)
    turn = []
    for a, b in zip(step[:-1], step[1:]):
        d = np.linalg.norm(a) * np.linalg.norm(b)
        turn.append(float(np.degrees(np.arccos(np.clip(a @ b / d, -1, 1)))) if d else 0.0)
    acc = np.linalg.norm(np.diff(step, axis=0), axis=1)
    return {
        "n_points": len(pos),
        "step_mm_per_frame": [round(float(s), 3) for s in speed],
        "turn_angle_deg": [round(t, 1) for t in turn],
        "accel_mm_per_frame2": [round(float(a), 3) for a in acc],
    }


def baseline(s: dict) -> bool:
    """Plain-code rule Jev has to beat: sharp turn or a step-size jump >2x."""
    st = s["step_mm_per_frame"]
    jump = max((max(a, b) / max(min(a, b), 1e-9) for a, b in zip(st, st[1:])), default=1)
    return max(s["turn_angle_deg"], default=0) > 90 or jump > 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zarr")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="triage.jsonl")
    args = ap.parse_args()

    trajs = [t for t in read_zarr_trajectories(args.zarr) if len(t.time()) >= 3]
    trajs = trajs[: args.limit]
    states = [summarise(t) for t in trajs]
    print(f"{len(trajs)} trajectories with >=3 points; baseline flags "
          f"{sum(baseline(s) for s in states)}")

    if args.dry_run:
        for s in states[:3]:
            print(json.dumps({"state": s, "question": QUESTION}, indent=1))
        return

    from typesafe_sdk import Noul, TypeSafeClient

    q = {"spurious": Noul(instructions=QUESTION, criteria=CRITERIA)}
    with TypeSafeClient() as client:
        def ask(s):
            return client.system_one(state={"trajectory": s}, questions=q)
        with ThreadPoolExecutor(8) as ex:
            resp = list(ex.map(ask, states))

    with open(args.out, "w") as f:
        for t, s, r in zip(trajs, states, resp):
            p = float(r.nouls["spurious"].noul)
            f.write(json.dumps({"trajid": int(t.trajid()), "p_spurious": p,
                                "baseline": baseline(s), **s}) + "\n")
    p = np.array([json.loads(l)["p_spurious"] for l in open(args.out)])
    b = np.array([baseline(s) for s in states])
    print(f"p>0.5: {(p > .5).sum()}  agree with baseline: {((p > .5) == b).mean():.0%}")
    print(f"wrote {args.out}; inspect disagreements by eye")


if __name__ == "__main__":
    main()
