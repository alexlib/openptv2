"""Measure a run and print the recommended two_phase tracking parameters.

    uv run python scripts/tracking_advice.py PATH_TO_RUN_FOLDER [--first 1] [--last 200] [--fps 5000]
    e.g. uv run python scripts/tracking_advice.py ~/data/experiment/wp2/test --fps 5000

The run folder holds parameters_*.yaml, cal/ and res/run.zarr (with correspondences and
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
    run = Path(a.run_dir).expanduser().resolve()
    if not run.is_dir():
        sys.exit(
            f"'{a.run_dir}' is not a folder. Give the path of a run folder, e.g.\n"
            "  uv run python scripts/tracking_advice.py "
            "~/Downloads/CompleteTest-e2e-local/wp2/test --fps 5000\n"
            "(RUN_DIR in the docs is a placeholder for your own path.)"
        )
    yamls = sorted(run.glob("parameters_*.yaml"))
    if not yamls:
        found = sorted(
            p.parent for p in run.glob("**/parameters_*.yaml") if "res" not in p.parts
        )[:8]
        hint = (
            "\nRun folders found below it:\n" + "\n".join(f"  {p}" for p in found)
            if found
            else ""
        )
        sys.exit(
            f"No parameters_*.yaml in {run}. A run folder contains parameters_*.yaml, "
            f"cal/ and res/run.zarr.{hint}"
        )
    if not (run / "res" / "run.zarr").exists():
        sys.exit(
            f"No res/run.zarr in {run}: run the sequence phase first (it creates the "
            "correspondences and targets this tool measures)."
        )
    yaml_path = yamls[0]
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
