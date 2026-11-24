# ruff: noqa
"""Tracker with the pixel-level de-merge on the rendered case syn1 (images + truth), scored against truth.

    POSTPTV_DIR=<clean postptv> uv run python bench/limits/syn1_pixel_fit.py <work dir> <n_frames> [variants]

variants (comma list): base, corr, corr_px (shared_correction with the pixel fit), px_only is not offered: the fit only
replaces the centre split inside the correction. Copies res/run.zarr, cal and the parameter file; the images are
symlinked (read only). Prints truth-based scores (common set via the base run's keys) and times.
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import synth_bench as sb  # noqa: E402

SRC = sb.SRC / "syn1" / "test"

VARIANTS = {
    "base": {},
    "corr": {"shared_correction": True},
    "corr_px": {"shared_correction": True, "holistic_pixel_fit": True},
    "corr_oracle": {"shared_correction": True, "holistic_pixel_fit": True},      # true pixel positions: the ceiling
}


class OracleFitter:
    """Stands in for the pixel fit with the truth projection (+ 0.05 px): what a perfect de-merge would give."""

    def __init__(self, noise_px=0.05):
        z = np.load(sb.SRC / "truth" / "truth.npz")
        self.fr, self.px = z["frame"], z["pixel_xy_per_cam"]
        self.by = {}
        self.rng = np.random.default_rng(0)
        self.noise = noise_px
        self.stats = {"groups": 0, "fitted": 0, "failed": 0, "noise": []}

    def __call__(self, frame, cam, centre, seeds, prior_px, others=None):
        from scipy.spatial import cKDTree

        if (frame, cam) not in self.by:
            m = self.fr == frame
            xy = self.px[m][:, 2 * cam: 2 * cam + 2]
            ok = np.isfinite(xy).all(1)
            self.by[(frame, cam)] = (xy[ok], cKDTree(xy[ok]))
        xy, tree = self.by[(frame, cam)]
        d, j = tree.query(seeds)
        pos = np.where((d < 1.0)[:, None], xy[j] + self.rng.normal(0, self.noise, seeds.shape), np.nan)
        var = np.where(d < 1.0, 0.01, np.nan)
        self.stats["groups"] += 1
        self.stats["fitted"] += 1
        return pos, var


def prepare(dst: Path, n_frames: int, over: dict) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "res").mkdir(parents=True)
    shutil.copytree(SRC / "cal", dst / "cal")
    shutil.copytree(SRC / "res" / "run.zarr", dst / "res" / "run.zarr")
    os.symlink(SRC / "img", dst / "img")
    raw = yaml.safe_load((SRC / "parameters_Run1.yaml").read_text())
    raw["sequence"]["first"], raw["sequence"]["last"] = 1, n_frames
    raw["track"].update(over)
    (dst / "parameters_Run1.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    return dst / "parameters_Run1.yaml"


def case_dir(work: Path, n_frames: int) -> Path:
    cd = work / "truth_case"
    cd.mkdir(parents=True, exist_ok=True)
    z = np.load(sb.SRC / "truth" / "truth.npz")
    keep = z["frame"] <= n_frames
    np.savez(cd / "truth.npz", particle_id=z["particle_id"][keep], frame=z["frame"][keep], pos_mm=z["pos_mm"][keep])
    (cd / "case.json").write_text(json.dumps({"k": 1, "n_out": 1500}))
    return cd


def run(work: Path, name: str, n_frames: int) -> dict:
    from openptv2.merge_post import arrays_from_store
    from openptv2.plugins import run_tracking_plugin
    from openptv2.storage import RunStore

    if name == "corr_oracle":
        import openptv2.holistic as H

        H.pixel_fitter_for = lambda exp: OracleFitter()
    dst = work / name
    yaml_path = prepare(dst, n_frames, VARIANTS[name])
    exp = sb._experiment(dst, yaml_path, 1, n_frames)
    cwd = os.getcwd()
    os.chdir(dst)
    try:
        t0 = time.perf_counter()
        run_tracking_plugin("two_phase", exp)
        t = time.perf_counter() - t0
    finally:
        os.chdir(cwd)
    st = RunStore(str(dst / "res" / "run.zarr"), mode="r")
    a = arrays_from_store(st, "ptv_is", 1, n_frames, 4)
    return {"time_s": t, "pred": {"trajid": a["tid"], "frame": a["frame"], "pos": np.asarray(a["pos"], float)}}


def main() -> None:
    work, n_frames = Path(sys.argv[1]), int(sys.argv[2])
    names = sys.argv[3].split(",") if len(sys.argv) > 3 else list(VARIANTS)
    work.mkdir(parents=True, exist_ok=True)
    cd = case_dir(work, n_frames)
    flow = sb.Flow()
    ref_keys = None
    out = {}
    for name in names:
        r = run(work, name, n_frames)
        ev = sb.evaluate_pred(cd, r["pred"], flow, ref_keys=ref_keys)
        if ref_keys is None and "_keys" in ev:
            ref_keys = ev["_keys"]
        ev.pop("_keys", None)
        ev["time_s"] = round(r["time_s"], 1)
        out[name] = ev
        keys = ("time_s", "vel_win_common", "pos_err_mm", "yield", "ghost_point_frac", "long_completeness", "long_fragments",
                "raw_link_precision", "out_points")
        print(name, {k: (round(ev[k], 4) if isinstance(ev.get(k), float) else ev.get(k)) for k in keys}, flush=True)
    (work / "result.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
