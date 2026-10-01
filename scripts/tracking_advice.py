"""Measure a run and print the recommended two_phase tracking parameters.

    uv run python scripts/tracking_advice.py RUN_DIR [--first 1] [--last 200] [--fps 5000]

RUN_DIR holds parameters_*.yaml, cal/ and res/run.zarr (with correspondences and
targets, i.e. after the sequence phase). Reads only; writes nothing. The numbers it
reports, and what to do with them, are explained in docs/tracking_parameters_guide.md.
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from openptv2.storage import RunStore  # noqa: E402
from openptv2.tracking_advice import advise, measure, report  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=200)
    ap.add_argument(
        "--fps", type=float, default=None, help="frame rate, for the window"
    )
    a = ap.parse_args(argv)
    run = Path(a.run_dir).resolve()
    yaml_path = next(run.glob("parameters_*.yaml"))
    track_cfg = (yaml.safe_load(yaml_path.read_text()) or {}).get("track", {})
    store = RunStore.open(run, mode="r")
    frames = sorted(int(k.split("_")[1]) for k in store.root["correspondences"].keys())
    frames = [f for f in frames if a.first <= f <= a.last]
    from openptv2.batch.pyptv_batch import build_processing_experiment

    cwd = os.getcwd()
    try:
        os.chdir(run)
        exp = build_processing_experiment(yaml_path, frames[0], frames[-1])
    finally:
        os.chdir(cwd)
    v_max = float(track_cfg.get("v_max", track_cfg.get("dvxmax", 1.0)))
    m = measure(store, frames, exp.cals, exp.cpar, v_max=v_max)
    print(report(m, advise(m, fps=a.fps)))
    pinned = [
        k
        for k in ("confirm_tol", "confirm_auto", "q_seed", "q_young")
        if k in track_cfg
    ]
    if pinned:
        print(
            "\nNOTE: your yaml already sets "
            + ", ".join(f"{k}: {track_cfg[k]}" for k in pinned)
        )


if __name__ == "__main__":
    main()
