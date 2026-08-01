# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "numpy>=2.0.0",
#     "matplotlib>=3.7.0",
#     "pyyaml>=6.0",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import numpy as np
    import yaml

    return mo, np, plt, patches, Path, yaml


@app.cell
def _(Path):
    base_path = Path("/home/user/Documents/GitHub/openptv2/test_data/test_cavity")
    res_dir = base_path / "res"
    res_dir.mkdir(exist_ok=True)
    img_dir = base_path / "img"
    return base_path, res_dir, img_dir


@app.cell
def _(mo):
    mo.md(
        """
        # Tracking Step Visualizer

        This notebook runs the **Python tracking engine** with an observer that
        records every per-particle decision: predicted position, search volume,
        candidates found, angle/acceleration scores, and the final link.

        Use the sliders below to explore individual particles at each frame.
        """
    )
    return


@app.cell
def _(base_path, res_dir, yaml):
    """Run tracking with the single runtime + observer."""
    import os
    import sys

    sys.path.insert(0, str(base_path.parent.parent))
    os.chdir(str(base_path))

    from openptv2.gui.pyptv import pyptv_batch

    yaml_path = base_path / "parameters_Run1.yaml"
    with open(yaml_path) as f:
        params = yaml.safe_load(f)

    params["sequence"]["output"] = str(res_dir)
    params["sequence"]["first"] = 10001
    params["sequence"]["last"] = 10004

    temp_yaml = base_path / "temp_run.yaml"
    with open(temp_yaml, "w") as f:
        yaml.dump(params, f)

    print("Running batch (correspondence): frames 10001-10004")
    pyptv_batch.main(temp_yaml, 10001, 10004)
    print("Batch complete!")
    return params, pyptv_batch, temp_yaml, yaml_path


@app.cell
def _(base_path):
    """Run tracking with the single runtime."""
    from openptv2.gui.pyptv.parameter_manager import ParameterManager
    from openptv2.gui.pyptv.ptv import py_start_proc_c

    from openptv2.tracker import Tracker, default_naming

    pm = ParameterManager(base_path / "parameters_Run1.yaml")
    cpar, spar, vpar, track_par, tpar, cals, epar = py_start_proc_c(pm)
    num_cams = cpar.get_num_cams()

    tracker = Tracker(cpar, vpar, track_par, spar, cals, default_naming)
    tracker.full_forward()

    print(
        f"Tracking complete: frames {spar.get_first()}–{spar.get_last()}, "
        f"{num_cams} cams."
    )
    # ponytail: per-particle observer not in Tracker API; add
    # TrackingObserver to track.py if step-level debug is needed.
    return tracker, cpar, cals, num_cams


@app.cell
def _(mo):
    mo.md(
        """
    > **Note**: Per-particle step visualization requires a
    > `TrackingObserver` callback in `openptv2.algorithms.track.Tracker`.
    > The `Tracker` currently runs `full_forward()` without per-step hooks.
    > Add a `TrackingObserver` class to `track.py` to re-enable the
    > particle-level debugger below.
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
