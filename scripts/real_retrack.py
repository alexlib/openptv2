"""Re-track a REAL run (default CompleteTest wp2) on a scratch copy with track overrides.

    uv run python scripts/real_retrack.py LABEL LAST [key=value ...]
    uv run python scripts/realism_metrics.py /private/tmp/claude-501/real_runs/LABEL/res/run.zarr --last LAST

Never writes into the source run. Example (A10 quality marks):
    uv run python scripts/real_retrack.py q 300 q_seed=0.2 q_young=3
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth_bench import _experiment

src = Path(
    os.environ.get(
        "REAL_RUN", Path.home() / "Downloads/CompleteTest-e2e-local/wp2/test"
    )
)
label = sys.argv[1]
last = int(sys.argv[2])
overrides = sys.argv[3:]
out = Path("/private/tmp/claude-501/real_runs") / label
if out.exists():
    shutil.rmtree(out)
(out / "res").mkdir(parents=True)
shutil.copytree(src / "cal", out / "cal")
shutil.copytree(src / "res" / "run.zarr", out / "res" / "run.zarr")
(out / "img").mkdir()
raw = yaml.safe_load((src / "parameters_Run1.yaml").read_text())
raw["sequence"]["last"] = last
for o in overrides:
    k, _, v = o.partition("=")
    raw["track"][k] = yaml.safe_load(v)
(out / "parameters_Run1.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
exp = _experiment(out, out / "parameters_Run1.yaml", 1, last)
from openptv2.plugins import run_tracking_plugin

os.chdir(out)
t0 = time.perf_counter()
run_tracking_plugin("two_phase", exp)
print("TRACK_TIME", round(time.perf_counter() - t0, 1))
