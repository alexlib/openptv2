"""Leave-one-camera-out residual (px) of 4-camera points: real recordings against the synthetic benchmark.

    uv run python bench/limits/loco_compare.py <label>=<run.zarr> [<label>=<run.zarr> ...] [--frames N]

The residual is how far the point triangulated from three cameras projects from the blob seen in the fourth: it
contains the blob-centre noise, the calibration error and the optics the model leaves out. If the real value is
several times the synthetic one, calibration, not tracking, limits the real data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from openptv2.loco import loco_residuals, summarize
from openptv2.recorrespond import load_parameters
from openptv2.storage import RunStore


def residuals(zarr_path: Path, params: Path, n_frames: int) -> np.ndarray:
    cpar, vpar, cals, _ = load_parameters(params)
    st = RunStore(zarr_path, mode="r")
    out = []
    for f in st.frames()[:n_frames:5]:
        pos, ids = st.read_correspondences(f)
        ids = np.asarray(ids)[:, :4]
        full = (ids >= 0).all(1)
        if not full.any():
            continue
        px = np.full((int(full.sum()), 4, 2), np.nan)
        for c in range(4):
            t = np.asarray(st.root[f"targets/cam_{c}/frame_{f:06d}"])
            px[:, c] = t[ids[full, c], 1:3]
        r = loco_residuals(px, cpar, cals, vpar)
        out.append(r[np.isfinite(r)])
    return np.concatenate(out)


def main() -> None:
    args = [a for a in sys.argv[1:] if "=" in a]
    n_frames = 200
    for a in args:
        label, _, path = a.partition("=")
        p = Path(path)
        params = p.parents[1] / "parameters_Run1.yaml"
        r = residuals(p, params, n_frames)
        s = summarize(r)
        print(f"{label:28s} n={s['n']:7d} median {s['median_px']:.3f} p90 {s['p90_px']:.3f} p99 {s['p99_px']:.3f} rms {s['rms_px']:.3f} px")


if __name__ == "__main__":
    main()
