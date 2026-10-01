"""Upper bound for any ghost-removal idea (plan A9-A11).

Copies a synthetic bench case as <case>_noghost with every ghost 3D point
(judged by the truth labels) removed from the correspondences. Tracking that
case shows how much error is due to ghosts alone:

    uv run python scripts/ghost_bound.py L1_d1_c0_k1_n200_s0.08
    uv run python scripts/synth_bench.py track \
        --case L1_d1_c0_k1_n200_s0.08_noghost --tracker two_phase
    uv run python scripts/synth_bench.py eval \
        --case L1_d1_c0_k1_n200_s0.08_noghost --trackers oracle two_phase
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth_bench import WORK, _match  # noqa: E402

from openptv2.storage import RunStore  # noqa: E402


def main(case: str) -> None:
    src = WORK / "cases" / case
    dst = WORK / "cases" / (case + "_noghost")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    tr = np.load(dst / "truth.npz")
    by_frame = {}
    for f in np.unique(tr["frame"]):
        m = tr["frame"] == f
        by_frame[int(f)] = (
            tr["pos_mm"][m],
            tr["particle_id"][m],
            cKDTree(tr["pos_mm"][m]),
        )
    store = RunStore(dst / "res" / "run.zarr", mode="a")
    n_out = int(json.loads((dst / "case.json").read_text())["n_out"])
    pos_l, cam_l, frm_l = [], [], []
    for f in range(1, n_out + 1):
        p, c = store.read_correspondences(f)
        pos_l.append(p)
        cam_l.append(c)
        frm_l.append(np.full(len(p), f))
    frame, pos = np.concatenate(frm_l), np.vstack(pos_l)
    g0, _, off = _match(by_frame, frame, pos, 0.6)
    bias = np.median(off[g0 >= 0], axis=0)
    gid, _, _ = _match(by_frame, frame, pos - bias, 0.4)
    k = dropped = 0
    for f in range(1, n_out + 1):
        n = len(pos_l[f - 1])
        keep = gid[k : k + n] >= 0
        k += n
        dropped += int((~keep).sum())
        store.write_correspondences(f, pos_l[f - 1][keep], cam_l[f - 1][keep])
    print(case, "ghosts dropped", dropped, "of", len(gid))


if __name__ == "__main__":
    main(sys.argv[1])
