"""How accurate are the corrected centres of merged blobs, in pixels, against truth?

    KEEP_STORE=1 uv run python scripts/synth_bench.py track --case <case> --tracker two_phase+shared_pass=1.7
    uv run python bench/limits/demerge_error.py <case> <label of that run>

Runs the refinement of the holistic loop (all rows, 3 passes) on the kept store of the run and compares, per
camera view, the pixel position the point uses with the projection of its true position. A per-camera offset
(the render and processing calibrations differ) is taken from the clean views and removed.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import synth_bench as sb  # noqa: E402

from openptv2.loco import project_pixels  # noqa: E402
from openptv2.merge_post import (  # noqa: E402
    arrays_from_store,
    estimate_signatures,
    extend_membership,
)
from openptv2.merge_refine import refine_merged, shared_blobs  # noqa: E402
from openptv2.recorrespond import load_parameters  # noqa: E402
from openptv2.storage import RunStore  # noqa: E402


def main() -> None:
    case, label = sys.argv[1], sys.argv[2]
    case_dir = sb.WORK / "cases" / case
    run_dir = sb.WORK / "runs" / case / label
    cpar, vpar, cals, _ = load_parameters(run_dir / "parameters_Run1.yaml")
    st = RunStore(run_dir / "res" / "run.zarr", mode="r")
    n_fr = len(st.frames())
    a = arrays_from_store(st, "ptv_is", 1, n_fr, 4)
    tr = np.load(case_dir / "truth.npz")
    blob, centre, sumg, _n = extend_membership(a["frame"], a["pos"], a["blob"], a["centre"], a["sumg"], a["blobs"],
                                               cpar, cals, radius=2.0)
    sig = estimate_signatures(a["tid"], a["frame"], blob, sumg)
    res = refine_merged(a["frame"], a["tid"], blob, centre, sumg, cpar, cals, vpar, signatures=sig, partner_min_len=10)
    share = shared_blobs(a["frame"], blob)

    # truth particle of every row: the one whose blob ids agree in the most cameras (at least 2)
    truth_px = np.full(centre.shape, np.nan)
    ok_row = np.zeros(len(a["frame"]), bool)
    for f in np.unique(a["frame"]):
        m = np.flatnonzero(tr["frame"] == f)
        bt, tp = tr["blob_tid"][m], tr["pos_mm"][m]
        owner = {}
        for k in range(len(m)):
            for c in range(4):
                if bt[k, c] >= 0:
                    owner.setdefault((c, int(bt[k, c])), []).append(k)
        for r in np.flatnonzero(a["frame"] == f):
            cnt = Counter()
            for c in range(4):
                if blob[r, c] >= 0:
                    for k in owner.get((c, int(blob[r, c])), []):
                        cnt[k] += 1
            if cnt and max(cnt.values()) >= 2:
                k = max(cnt, key=cnt.get)
                ok_row[r] = True
                for c in range(4):
                    if blob[r, c] >= 0:
                        truth_px[r, c] = project_pixels(tp[k][None], cals[c], cpar)[0]
    print(f"rows with a truth particle: {ok_row.mean():.3f}")
    for c in range(4):                                   # per-camera calibration offset from clean views
        cl = ok_row & (share[:, c] == 1) & np.isfinite(truth_px[:, c, 0])
        off = np.median(centre[cl, c] - truth_px[cl, c], axis=0)
        truth_px[:, c] += off
    obs = res.obs_px
    groups = {"clean views": share == 1, "shared views (2 particles)": share == 2, "shared views (3+)": share >= 3}
    print(f"{'':28s} {'n':>8s}  raw centre px: median rms | corrected px: median rms   (gated out)")
    for name, g in groups.items():
        m = g & ok_row[:, None] & np.isfinite(truth_px[:, :, 0])
        raw = np.linalg.norm(centre[m] - truth_px[m], axis=1)
        cor_m = m & np.isfinite(obs[:, :, 0])
        cor = np.linalg.norm(obs[cor_m] - truth_px[cor_m], axis=1)
        print(f"{name:28s} {int(m.sum()):8d}  {np.median(raw):6.3f} {np.sqrt(np.mean(raw**2)):6.3f} |"
              f" {np.median(cor):6.3f} {np.sqrt(np.mean(cor**2)):6.3f}   ({int(m.sum() - cor_m.sum())})")
        if name.startswith("shared views (2"):
            sd = np.sqrt(res.obs_var[cor_m])
            z = cor / sd
            q = lambda v: " ".join(f"{np.percentile(v, p_):.2f}" for p_ in (50, 75, 90, 95, 99))  # noqa: E731
            print(f"    percentiles 50/75/90/95/99   raw {q(raw)}   corrected {q(cor)}")
            print(f"    share above 1 px: raw {np.mean(raw > 1):.3f}  corrected {np.mean(cor > 1):.3f}; "
                  f"reported sd median {np.median(sd):.2f} px, error/sd: median {np.median(z):.2f} p90 {np.percentile(z, 90):.2f} "
                  f"p99 {np.percentile(z, 99):.2f}")
            # does the reported variance flag the bad ones?
            bad = cor > 1.0
            print(f"    reported sd of the >1 px errors: median {np.median(sd[bad]):.2f} px (all: {np.median(sd):.2f})")
            # by what the correction rested on: a trusted prediction of the point itself
            has_pred = np.isfinite(res.pred_px[cor_m][:, 0])
            print(f"    point has a trusted prediction: {has_pred.mean():.2f}; median error with / without: "
                  f"{np.median(cor[has_pred]):.3f} / {np.median(cor[~has_pred]) if (~has_pred).any() else float('nan'):.3f}")


if __name__ == "__main__":
    main()
