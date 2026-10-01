"""A10 after tracking, on a synthetic bench run: trim doubtful ends, drop spikes,
weight points by quality; writes new runs that `synth_bench.py eval` can score.

    uv run python scripts/quality_postprocess.py CASE RUN_LABEL SPEC [SPEC ...]

SPEC is a '+'-free string of parts joined by '_':
    unit            unit weights through the weighted smoother (control: must equal the
                    flowtracks result)
    wP              weights (1 - ghost) ** P            e.g. w1, w2
    trimT           cut up to 3 doubtful end points with ghost probability > T   e.g. trim0.5
    spikeT:J        drop interior points with ghost > T that exceed the mean of their
                    two neighbours by more than J      e.g. spike0.3:0.3
Output run: <RUN_LABEL>~<SPEC>, e.g. two_phase+q_seed=0.2+q_young=3~trim0.5_w2
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth_bench import WORK, _experiment  # noqa: E402

from openptv2.point_quality import (  # noqa: E402
    fit_scale,
    frame_rcm,
    ghost_probability,
)
from openptv2.quality_post import (  # noqa: E402
    drop_spikes,
    quality_weights,
    trim_doubtful_ends,
)
from openptv2.storage import RunStore  # noqa: E402


def point_ghost_probability(case: str, pred: dict) -> np.ndarray:
    cd = WORK / "cases" / case
    n_out = int(json.loads((cd / "case.json").read_text())["n_out"])
    exp = _experiment(cd, cd / "parameters_Run1.yaml", 1, n_out)
    store = RunStore.open(cd, mode="r")
    frames = list(range(1, n_out + 1))
    pos, rcm, nseen = [], [], []
    for f in frames:
        p, c = store.read_correspondences(f)
        pos.append(p)
        rcm.append(frame_rcm(store, f, exp.cals, exp.cpar))
        nseen.append((c >= 0).sum(axis=1))
    scale = fit_scale(np.vstack(pos), np.concatenate(rcm), np.concatenate(nseen))
    pg = [ghost_probability(p, r, n, scale) for p, r, n in zip(pos, rcm, nseen)]
    out = np.full(len(pred["frame"]), 0.5)
    for f in np.unique(pred["frame"]):
        m = pred["frame"] == f
        d, j = cKDTree(pos[int(f) - 1]).query(pred["pos"][m])
        out[m] = np.where(d < 1e-6, pg[int(f) - 1][j], 0.5)
    return out


def apply_spec(pred: dict, pg: np.ndarray, spec: str) -> dict:
    tid, fr, pos = pred["trajid"], pred["frame"], pred["pos"]
    keep = np.ones(len(tid), bool)
    weights = None
    for part in spec.split("_"):
        if part == "unit":
            weights = np.ones(len(tid))
        elif part.startswith("trim"):
            keep &= trim_doubtful_ends(tid, fr, pg, float(part[4:]), 3)
        elif part.startswith("spike"):
            t, j = part[5:].split(":")
            keep &= drop_spikes(tid, fr, pg, float(t), float(j))
        elif part.startswith("w"):
            weights = quality_weights(pg, float(part[1:]))
        else:
            raise ValueError(part)
    out = {"trajid": tid[keep], "frame": fr[keep], "pos": pos[keep]}
    if weights is not None:
        out["w"] = weights[keep]
    return out


def main(case: str, label: str, specs: list[str]) -> None:
    pred = dict(np.load(WORK / "runs" / case / label / "pred.npz"))
    pg = point_ghost_probability(case, pred)
    for spec in specs:
        res = apply_spec(pred, pg, spec)
        name = f"{label}~{spec}"
        run = WORK / "runs" / case / name
        run.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(run / "pred.npz", **res)
        (run / "run.json").write_text(
            json.dumps({"tracker": name, "case": case, "time_s": 0, "status": "ok"})
        )
        print(f"{name}: kept {len(res['frame'])}/{len(pred['frame'])} points")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
