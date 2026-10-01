"""A5 on a synthetic bench run: reconnect broken pieces; writes runs `synth_bench.py eval` scores.

    uv run python scripts/reconnect_post.py CASE RUN_LABEL SPEC [SPEC ...]

SPEC: rcG_T[_nN][_dD]  G = largest frame step joined, T = tolerance in noise sigmas,
N = points fitted per side (10), D = polynomial degree (2). e.g. rc4_4  rc3_6_n8_d1
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth_bench import WORK  # noqa: E402

from openptv2.reconnect import reconnect_pieces  # noqa: E402


def main(case, label, specs):
    pred = dict(np.load(WORK / "runs" / case / label / "pred.npz"))
    for spec in specs:
        parts = spec.split("_")
        kw = dict(max_gap=int(parts[0][2:]), tol_sigmas=float(parts[1]))
        for p in parts[2:]:
            if p[0] == "n":
                kw["n_fit"] = int(p[1:])
            elif p[0] == "d":
                kw["deg"] = int(p[1:])
        new = reconnect_pieces(pred["trajid"], pred["frame"], pred["pos"], **kw)
        res = {**pred, "trajid": new}
        name = f"{label}~{spec}"
        run = WORK / "runs" / case / name
        run.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(run / "pred.npz", **res)
        (run / "run.json").write_text(
            json.dumps({"tracker": name, "case": case, "time_s": 0, "status": "ok"})
        )
        print(
            f"{name}: tracks {len(np.unique(pred['trajid']))} -> {len(np.unique(new))}"
        )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
