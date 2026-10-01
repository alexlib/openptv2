"""A9 on a synthetic bench run: drop points that do not fit the curve their neighbours define.

    uv run python scripts/smoothness_post.py CASE RUN_LABEL SPEC [SPEC ...]

SPEC: smK[_wW][_dD][_sN]  K = threshold in median residuals, W = window points (11),
D = polynomial degree (2), N = drop whole trajectories shorter than N points (0).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth_bench import WORK  # noqa: E402

from openptv2.smoothness_filter import smoothness_keep  # noqa: E402


def main(case, label, specs):
    pred = dict(np.load(WORK / "runs" / case / label / "pred.npz"))
    for spec in specs:
        parts = spec.split("_")
        kw = dict(k=float(parts[0][2:]))
        for p in parts[1:]:
            if p[0] == "w":
                kw["window"] = int(p[1:])
            elif p[0] == "d":
                kw["deg"] = int(p[1:])
            elif p[0] == "s":
                kw["min_keep_len"] = int(p[1:])
        keep = smoothness_keep(pred["trajid"], pred["frame"], pred["pos"], **kw)
        res = {k: v[keep] for k, v in pred.items() if len(v) == len(keep)}
        name = f"{label}~{spec}"
        run = WORK / "runs" / case / name
        run.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(run / "pred.npz", **res)
        (run / "run.json").write_text(
            json.dumps({"tracker": name, "case": case, "time_s": 0, "status": "ok"})
        )
        print(f"{name}: kept {keep.sum()}/{len(keep)} points")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
