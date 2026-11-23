"""Truth-free check of the shared-blob second pass on a real recording (scratch copy; the source is only read).

    uv run python bench/real_check/shared_pass_real.py <source test dir> <work dir> <n_frames> [ratio]

Copies ``res/run.zarr``, ``cal`` and the parameter file of a run folder (e.g. CompleteTest-e2e-local/wp2/test) to
``<work dir>/{base,shared}``, matches the first ``n_frames`` frames again with and without ``ptv.shared_pass``,
tracks both with the run's own ``track`` settings, and prints track statistics: points per frame, track lengths,
roughness (second difference of the position along a track) of ordinary and of new points, singleton share.
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

oc = importlib.import_module("openptv2.correspondences")


def prepare(src: Path, dst: Path, n_frames: int) -> Path:
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "res").mkdir(parents=True)
    shutil.copytree(src / "cal", dst / "cal")
    shutil.copytree(src / "res" / "run.zarr", dst / "res" / "run.zarr")
    raw = yaml.safe_load((src / "parameters_Run1.yaml").read_text())
    raw["sequence"]["first"], raw["sequence"]["last"] = 1, n_frames
    (dst / "parameters_Run1.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    return dst / "parameters_Run1.yaml"


def run_variant(src: Path, dst: Path, n_frames: int, shared_pass) -> dict:
    from openptv2.plugins import run_tracking_plugin
    from openptv2.storage import RunStore

    yaml_path = prepare(src, dst, n_frames)
    exp = sb._experiment(dst, yaml_path, 1, n_frames)
    t0 = time.perf_counter()
    oc.match_correspondences_batch_parallel(
        frames=range(1, n_frames + 1), cpar=exp.cpar, cals=exp.cals, vpar=exp.vpar,
        zarr_store_path=str(dst / "res" / "run.zarr"), n_workers=8, write_to_store=True, shared_pass=shared_pass,
    )
    t_match = time.perf_counter() - t0
    exp = sb._experiment(dst, yaml_path, 1, n_frames)
    cwd = os.getcwd()
    os.chdir(dst)
    try:
        t1 = time.perf_counter()
        run_tracking_plugin("two_phase", exp)
        t_track = time.perf_counter() - t1
    finally:
        os.chdir(cwd)
    return {"t_match": t_match, "t_track": t_track, "store": str(dst / "res" / "run.zarr"), "n_frames": n_frames,
            "_open": RunStore}


def stats(info: dict, n_base: dict | None) -> tuple[dict, dict]:
    from openptv2.merge_post import arrays_from_store

    st = info["_open"](info["store"], mode="r")
    a = arrays_from_store(st, "ptv_is", 1, info["n_frames"], 4)
    frame, row, pos, tid = a["frame"], a["row"], a["pos"], a["tid"]
    n = len(frame)
    new = np.zeros(n, bool)
    if n_base is not None:
        new = row >= np.array([n_base[int(f)] for f in frame])
    o = np.lexsort((frame, tid))
    t_s, f_s, p_s = tid[o], frame[o], pos[o]
    length = np.bincount(np.unique(t_s, return_inverse=True)[1])
    inv = np.unique(t_s, return_inverse=True)[1]
    plen = length[inv]
    # roughness: |x_{t-1} - 2 x_t + x_{t+1}| of interior points of one track with consecutive frames
    ok = (t_s[:-2] == t_s[2:]) & (f_s[2:] - f_s[:-2] == 2)
    rough = np.full(n, np.nan)
    rough[o[1:-1][ok]] = np.linalg.norm(p_s[:-2][ok] - 2 * p_s[1:-1][ok] + p_s[2:][ok], axis=1)
    kinds = {"all": np.ones(n, bool), "ordinary": ~new, "new": new}
    out = {
        "points": n, "points_per_frame": n / info["n_frames"], "tracks": int(len(length)),
        "mean_len": float(length.mean()), "tracks_ge_50": int((length >= 50).sum()),
        "pts_in_tracks_ge_50": float((plen >= 50).mean()), "pts_in_tracks_ge_200": float((plen >= 200).mean()),
        "singletons": float((plen == 1).mean()), "short_lt5": float((plen < 5).mean()),
        "new_points": int(new.sum()), "t_match_s": round(info["t_match"], 1), "t_track_s": round(info["t_track"], 1),
    }
    for k, m in kinds.items():
        r = rough[m & np.isfinite(rough)]
        if len(r):
            out[f"rough_{k}_median"] = float(np.median(r))
            out[f"rough_{k}_p90"] = float(np.percentile(r, 90))
    if new.any():
        out["new_in_tracks_ge_5"] = float((plen[new] >= 5).mean())
        out["new_in_tracks_ge_50"] = float((plen[new] >= 50).mean())
    return out, {int(f): int((frame == f).sum()) for f in np.unique(frame)}


def main() -> None:
    src, work, n_frames = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    ratio = float(sys.argv[4]) if len(sys.argv) > 4 else oc.DEFAULT_SHARED_PASS
    base = run_variant(src, work / "base", n_frames, None)
    base_stats, _ = stats(base, None)
    # number of standard rows per frame = the rows of the base run's correspondences
    st = base["_open"](base["store"], mode="r")
    n_base = {f: len(st.read_correspondences(f)[0]) for f in range(1, n_frames + 1)}
    sh = run_variant(src, work / "shared", n_frames, ratio)
    sh_stats, _ = stats(sh, n_base)
    res = {"base": base_stats, "shared": sh_stats, "ratio": ratio}
    (work / "result.json").write_text(json.dumps(res, indent=2))
    keys = sorted(set(base_stats) | set(sh_stats))
    print(f"{'':28s} {'base':>12s} {'shared':>12s}")
    for k in keys:
        b, s = base_stats.get(k), sh_stats.get(k)
        fmt = lambda v: "-" if v is None else (f"{v:12.4f}" if isinstance(v, float) else f"{v:12d}")  # noqa: E731
        print(f"{k:28s} {fmt(b):>12s} {fmt(s):>12s}")


if __name__ == "__main__":
    main()
