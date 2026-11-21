"""Downstream score of a tracker: run the openptv-cloud post-processing (reconnect, repair, smoothing) and the
openptv-analysis Eulerian stages on a scratch copy of an experiment whose ``res/run.zarr`` was replaced, then compare the
phase/region metrics table (mean velocity, MKE, VSS, eps) with a baseline run and with the legacy (MATLAB) table.

    uv run python scripts/eulerian_compare.py BASE_EXPERIMENT NEW_LABEL wp1/test=<run.zarr> wp2/test=<run.zarr>

``BASE_EXPERIMENT``: a scratch experiment folder (experiment.yaml, roi.json, wpN/test/{cal,parameters,res/run.zarr});
``NEW_LABEL``: its copy is made next to it with the given run stores swapped in.  Never point it at the real data.

Environment notes (2026-11): the post step must run in the openptv-cloud *tool* environment (flowtracks 1.6.1; other
versions write zero velocities), the analysis stages in the openptv-analysis project with its flowtracks zarr reader
disabled (it returns nothing there); the filter stage needs no ``legacy_root`` (remove it from the scratch yaml) and
``postprocess`` must be skipped (it rewrites ``trajectories/`` with zero velocities).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

TOOL_PY = Path.home() / ".local/share/uv/tools/openptv-cloud/bin/python"
ANALYSIS = Path.home() / "Documents/Github/openptv-analysis"
LEGACY = Path.home() / "Downloads/HiDImaging/CompleteTest-e2e-legacy/outputs/run_all/Results_Tables/phaseRegionMetrics.csv"
POST = "import sys; from openptv_cloud.post import convert; convert(sys.argv[1], 1, 1072, 5000.0, 5)"
PIPE = """
import sys
import openptv_analysis.io as io
io._ft_read_zarr = None
from openptv_analysis.cli import main
sys.argv = ["openptv-pipeline", "run", sys.argv[1], "--from-stage", "filter"]
main()
"""
COLS = ["vel_mean_global", "MKE_mean_global", "VSS_mean_global", "eps_mean_global",
        "vel_mean_jet", "VSS_mean_jet", "eps_mean_jet"]


def main() -> None:
    base, label, *swaps = sys.argv[1:]
    base = Path(base).resolve()
    new = base.parent / label
    if new.exists():
        shutil.rmtree(new)
    shutil.copytree(base, new, ignore=shutil.ignore_patterns("outputs", "ensemble", "vtk_output"))
    runs = []
    for s in swaps:
        run, src = s.split("=")
        runs.append(run)
        dst = new / run / "res" / "run.zarr"
        shutil.rmtree(dst)
        shutil.copytree(src, dst)
    for run in runs:
        subprocess.run([str(TOOL_PY), "-c", POST, str(new / run)], check=True)
    subprocess.run(["uv", "run", "python", "-c", PIPE, str(new)], cwd=ANALYSIS, check=False)
    import pandas as pd

    a = pd.read_csv(base / "outputs/run_all/Results_Tables/phaseRegionMetrics.csv").set_index("Phase")
    b = pd.read_csv(new / "outputs/run_all/Results_Tables/phaseRegionMetrics.csv").set_index("Phase")
    ref = pd.read_csv(LEGACY).set_index("Phase") if LEGACY.exists() else None
    rows = a.dropna(subset=["vel_mean_global"]).index
    for c in COLS:
        print(f"\n{c}")
        t = pd.DataFrame({"baseline": a.loc[rows, c], label: b.loc[rows, c]})
        if ref is not None:
            t["legacy"] = ref.loc[rows, c]
        print(t.round(4).to_string())


if __name__ == "__main__":
    main()
