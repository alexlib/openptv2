"""Tracker benchmark on the Synthetic-CompleteTest ground truth.

"Best" = after tracking, postptv gap repair and Savitzky-Golay smoothing, the
measured trajectories are closest to the TRUE fluid flow: no ghost tracers,
long trajectories kept whole, accurate velocity/acceleration. Then density
(how that holds as particles get denser/clustered), then speed.

Cases are built from the real measured flow field (truth/flow_field.npz) and
the real calibration of the synthetic run, at three input levels:

  L0  truth 3D positions + exact 2D projections, identity correspondences
      (pure linking test: every miss is the tracker's fault)
  L1  2D targets synthesised like the rendered images would give them
      (render-calibration error, centroid noise, dim-particle dropout,
      overlapping particles merged into one blob), then the REAL openptv2
      correspondence code -> real ghosts and real 3D noise
  L2  the real image pipeline output already in syn1/test/res/run.zarr
      (native density only), optionally frame-subsampled

Stress axes: particle density multiplier, inhomogeneous clustering
(seeding weighted by velocity-gradient magnitude + continuous "injector"
blobs in the highest-shear regions that the flow stretches into dense
filaments), and frame skip k (dt = k / 5000 s).

Nothing in the source case folder is modified; every case, tracker run and
result lives under WORK.

Usage (repo root):
    uv run python scripts/synth_bench.py build --level L1 --mult 1 --k 1
    uv run python scripts/synth_bench.py track --case <case> --tracker two_phase
    uv run python scripts/synth_bench.py eval --case <case>
    uv run python scripts/synth_bench.py sweep --cases a b --trackers x y -j 6
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

# postptv (flowtracks) from the local checkout: the gap repair
# (flowtracks.repair) is newer than the flowtracks release installed here
POSTPTV = Path("/Users/alex/Documents/Github/postptv")
if POSTPTV.exists():
    sys.path.insert(0, str(POSTPTV))

SRC = Path("/Users/alex/Downloads/HiDImaging/Synthetic-CompleteTest")
SRC_RUN = SRC / "syn1" / "test"
WORK = Path("/Users/alex/Downloads/HiDImaging/tracker-bench-2026-09-29")
FPS = 5000.0
NUM_CAMS = 4

TRACKERS = [
    "priority_segment_3d",
    "4be",
    "trackcorr",
    "full_multipass",
    "two_directional",
    "nearest_hungarian_3d",
    "predictive_gmm_3d",
    "hybrid_deltat_3d",
    "myptv_2d_tracking",
    "two_phase",
]


# --------------------------------------------------------------------------
# truth flow field (same maths as openptv_analysis.synthetic_experiment)
# --------------------------------------------------------------------------


class Flow:
    """Truth velocity field, mm/frame at the native 5000 fps."""

    def __init__(self, path: Path = SRC / "truth" / "flow_field.npz"):
        d = np.load(path)
        self.lo, self.voxel, self.mask = d["lo"], float(d["voxel"]), d["mask"]
        self.density, self.tc, self.u = d["density"], d["t_centres"], d["u"]
        self.shape = self.mask.shape
        axes = [
            self.lo[k] + self.voxel * (np.arange(self.shape[k]) + 0.5)
            for k in range(3)
        ]
        self.fields = [
            RegularGridInterpolator(axes, uk, bounds_error=False, fill_value=0.0)
            for uk in self.u
        ]

    def velocity(self, p, t):
        tc = self.tc
        if t <= tc[0]:
            return self.fields[0](p)
        if t >= tc[-1]:
            return self.fields[-1](p)
        k = int(np.searchsorted(tc, t)) - 1
        a = (t - tc[k]) / (tc[k + 1] - tc[k])
        return (1 - a) * self.fields[k](p) + a * self.fields[k + 1](p)

    def material_acc(self, p, t):
        """Du/Dt in mm/frame^2, same central form as the renderer."""
        v = self.velocity(p, t)
        return 0.5 * (self.velocity(p + v, t + 1.0) - self.velocity(p - v, t - 1.0))

    def inside(self, p):
        idx = np.floor((p - self.lo) / self.voxel).astype(int)
        ok = np.all((idx >= 0) & (idx < self.shape), axis=1)
        out = np.zeros(len(p), bool)
        out[ok] = self.mask[tuple(idx[ok].T)]
        return out

    def shear(self):
        """|grad u| (Frobenius, 1/frame) on the voxel grid, time-averaged."""
        u = self.u.mean(axis=0)
        g2 = np.zeros(self.shape)
        for c in range(3):
            for ax in range(3):
                g2 += np.gradient(u[..., c], self.voxel, axis=ax) ** 2
        s = np.sqrt(g2)
        s[~self.mask] = 0.0
        return s

    def seed_from(self, w, n, rng):
        w = w.ravel()
        cells = rng.choice(len(w), n, p=w / w.sum())
        idx = np.column_stack(np.unravel_index(cells, self.shape))
        return self.lo + self.voxel * (idx + rng.random((n, 3)))


def _advect_truth(flow: Flow, n: int, n_out: int, k: int, cluster: float, seed: int):
    """Advect n particles (RK4, native dt), keep every k-th frame.

    cluster in [0, 1): fraction of particles that are injector-blob particles
    (seeded, and re-seeded when they leave, in Gaussian blobs at the highest-
    shear voxels); the rest are seeded with density x (1 + 4 shear/shear_99)
    when cluster > 0, else with the real density alone.
    """
    rng = np.random.default_rng(seed)
    w = flow.density.copy()
    centres = np.zeros((0, 3))
    if cluster > 0:
        s = flow.shear()
        w = w * (1.0 + 4.0 * s / np.percentile(s[flow.mask], 99))
        order = np.argsort(s.ravel())[::-1]
        picked: list[np.ndarray] = []
        for c in order:
            ijk = np.array(np.unravel_index(c, flow.shape))
            if all(np.abs(ijk - q).max() >= 6 for q in picked):
                picked.append(ijk)
            if len(picked) == 6:
                break
        centres = flow.lo + flow.voxel * (np.array(picked) + 0.5)

    def seed_blob(m):
        out = np.empty((0, 3))
        while len(out) < m:
            c = centres[rng.integers(len(centres), size=2 * m)]
            p = c + rng.normal(0, 1.0, (2 * m, 3))
            out = np.vstack([out, p[flow.inside(p)]])
        return out[:m]

    n_blob = int(round(cluster * n))
    pos = np.vstack([flow.seed_from(w, n - n_blob, rng), seed_blob(n_blob)])
    blob = np.r_[np.zeros(n - n_blob, bool), np.ones(n_blob, bool)]
    pid = np.arange(n)
    next_id = n

    def rk4(p, t):
        k1 = flow.velocity(p, t)
        k2 = flow.velocity(p + 0.5 * k1, t + 0.5)
        k3 = flow.velocity(p + 0.5 * k2, t + 0.5)
        k4 = flow.velocity(p + k3, t + 1.0)
        return p + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    rows = {k_: [] for k_ in ("id", "frame", "pos", "blob")}
    n_native = 1 + (n_out - 1) * k
    for f in range(1, n_native + 1):
        if (f - 1) % k == 0:
            rows["id"].append(pid.copy())
            rows["frame"].append(np.full(n, 1 + (f - 1) // k))
            rows["pos"].append(pos.copy())
            rows["blob"].append(blob.copy())
        pos = rk4(pos, float(f))
        gone = ~flow.inside(pos)
        if gone.any():
            gb, gf = gone & blob, gone & ~blob
            if gf.any():
                pos[gf] = flow.seed_from(w, int(gf.sum()), rng)
            if gb.any():
                pos[gb] = seed_blob(int(gb.sum()))
            m = int(gone.sum())
            pid[gone] = np.arange(next_id, next_id + m)
            next_id += m
    ids = np.concatenate(rows["id"])
    # per-particle brightness, log-normal as in the renderer
    amp = 150.0 * np.exp(np.random.default_rng(seed + 1).normal(0, 1.1, next_id))
    return {
        "particle_id": ids,
        "frame": np.concatenate(rows["frame"]),
        "pos_mm": np.concatenate(rows["pos"]),
        "blob": np.concatenate(rows["blob"]),
        "amp": amp[ids],
    }


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def _experiment(run_dir: Path, yaml_path: Path, first: int, last: int):
    from openptv2.batch.pyptv_batch import build_processing_experiment

    cwd = os.getcwd()
    try:
        os.chdir(run_dir)
        return build_processing_experiment(yaml_path, first, last)
    finally:
        os.chdir(cwd)


def _project(pts, cal, cpar):
    from openptv2.algorithms.imgcoord import img_coord_batch

    m = np.asarray(img_coord_batch(np.asarray(pts, float), cal, cpar.mm), float)
    return np.column_stack(
        [m[:, 0] / cpar.pix_x + cpar.imx / 2, cpar.imy / 2 - m[:, 1] / cpar.pix_y]
    )


def _render_cals(cals, cpar):
    """Recover the calibration the images were rendered with: per camera, the
    position/angle offsets from the processing calibration that reproduce the
    truth pixel coordinates (truth.npz) -- so L1 carries exactly the image
    case's calibration error. Cached in WORK."""
    from scipy.optimize import least_squares

    cache = WORK / "render_cal_offsets.npy"
    z = np.load(SRC / "truth" / "truth.npz")
    m = np.isin(z["frame"], [1, 200, 400])
    pts, px = z["pos_mm"][m], z["pixel_xy_per_cam"][m]
    if cache.exists():
        offs = np.load(cache)
    else:
        offs = np.zeros((len(cals), 6))
        for c, cal in enumerate(cals):
            ok = ~np.isnan(px[:, 2 * c])
            p0, a0 = np.asarray(cal.get_pos(), float), np.asarray(cal.get_angles(), float)

            def resid(x, c=c, cal=cal, ok=ok, p0=p0, a0=a0):
                cal.set_pos(p0 + x[:3])
                cal.set_angles(a0 + x[3:] * 1e-3)
                return (_project(pts[ok], cal, cpar) - px[ok, 2 * c: 2 * c + 2]).ravel()

            offs[c] = least_squares(resid, np.zeros(6), x_scale=0.01).x
            cal.set_pos(p0)
            cal.set_angles(a0)
        np.save(cache, offs)
    for cal, x in zip(cals, offs):
        cal.set_pos(np.asarray(cal.get_pos(), float) + x[:3])
        cal.set_angles(np.asarray(cal.get_angles(), float) + x[3:] * 1e-3)
    return cals


# --------------------------------------------------------------------------
# case building
# --------------------------------------------------------------------------


DEFAULT_SIGMA_PX = 0.04


def case_name(level, mult, cluster, k, n_out, sigma_px=DEFAULT_SIGMA_PX):
    """sigma_px (blob-centre noise) is part of the name only when changed, so
    the existing case names stay valid; e.g. ..._s0.06 is the real-jitter case."""
    tag = "" if sigma_px == DEFAULT_SIGMA_PX else f"_s{sigma_px:g}"
    return f"{level}_d{mult:g}_c{cluster:g}_k{k}_n{n_out}{tag}"


def _kinematic_bounds(truth: dict, k: int, noise_mm: float) -> dict:
    """Same physical search bounds for every tracker, from the truth: the
    99.9th percentile per-axis displacement per output frame x 1.5, and the
    same percentile of the change of displacement x 2 for dacc -- each widened
    by 3 sigma of the 3D position noise (noise_mm, depth axis) as it enters a
    first (sqrt 2) or second (sqrt 6) difference."""
    pid, fr, p = truth["particle_id"], truth["frame"], truth["pos_mm"]
    o = np.lexsort((fr, pid))
    pid, fr, p = pid[o], fr[o], p[o]
    same = (pid[1:] == pid[:-1]) & (fr[1:] == fr[:-1] + 1)
    d = (p[1:] - p[:-1])[same]
    dv = float(np.percentile(np.abs(d), 99.9)) * 1.5
    same2 = same[1:] & same[:-1]
    dd = (p[2:] - 2 * p[1:-1] + p[:-2])[same2]
    da = float(np.percentile(np.linalg.norm(dd, axis=1), 99.9)) * 2.0
    vmax = float(np.percentile(np.linalg.norm(d, axis=1), 99.9)) * 1.5
    n1, n2 = float(3 * np.sqrt(2) * noise_mm), float(3 * np.sqrt(6) * noise_mm)
    return {"dv": round(dv + n1, 4), "dacc": round(max(da + n2, 0.05), 4),
            "v_max": round(vmax + n1, 4), "noise_mm": noise_mm}


def _write_yaml(case_dir: Path, n_out: int, bounds: dict) -> Path:
    raw = yaml.safe_load((SRC_RUN / "parameters_Run1.yaml").read_text())
    raw["sequence"]["first"], raw["sequence"]["last"] = 1, n_out
    raw.setdefault("pft_version", {})["Existing_Target"] = 1
    raw["ptv"]["parallel_preprocess"] = False
    tr = raw["track"]
    for ax in "xyz":
        tr[f"dv{ax}min"], tr[f"dv{ax}max"] = -bounds["dv"], bounds["dv"]
    tr["dacc"] = bounds["dacc"]
    tr["v_max"] = bounds["v_max"]
    tr["postprocess"] = False
    out = case_dir / "parameters_Run1.yaml"
    out.write_text(yaml.safe_dump(raw, sort_keys=False))
    return out


def _copy_cal(case_dir: Path) -> None:
    (case_dir / "cal").mkdir(parents=True, exist_ok=True)
    for f in (SRC_RUN / "cal").iterdir():
        if f.suffix in (".ori", ".addpar") or f.name == "atrium_calblock_new.txt":
            shutil.copy2(f, case_dir / "cal" / f.name)
    (case_dir / "img").mkdir(exist_ok=True)


def _target_rows(xy, amp, merge, sigma_px, rng, extra=None, amp_min=0.0):
    """One camera: (N,2) pixels -> detected-target rows [pnr,x,y,n,nx,ny,sumg,tnr]
    plus, per input particle, its target index (-1 = lost).

    merge: two spots whose discs touch become one brightness-weighted target
    (disc radius psf*sqrt(2 ln(amp/40)); 40 grey levels, not targ_rec's 10,
    because its discontinuity split separates touching peaks above that).
    A (merged) spot dimmer than amp_min in total is not detected. With
    amp_min=49 this reproduces the image pipeline's 2D recall per
    nearest-neighbour distance bin within ~2% (0.746 overall vs 0.762).
    extra: (M, 8) rows of false targets (static background) appended before
    y-sorting."""
    n = len(xy)
    parent = np.arange(n)
    if merge and n > 1:
        rad = 0.72 * np.sqrt(2 * np.log(np.maximum(amp / 40.0, 1.0)))
        pairs = cKDTree(xy).query_pairs(2 * rad.max(), output_type="ndarray")
        d = np.linalg.norm(xy[pairs[:, 0]] - xy[pairs[:, 1]], axis=1)
        pairs = pairs[d < rad[pairs[:, 0]] + rad[pairs[:, 1]]]

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for a, b in pairs:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        parent = np.array([find(i) for i in range(n)])
    roots, inv = np.unique(parent, return_inverse=True)
    w = np.bincount(inv, weights=amp)
    cx = np.bincount(inv, weights=amp * xy[:, 0]) / w
    cy = np.bincount(inv, weights=amp * xy[:, 1]) / w
    cnt = np.bincount(inv)
    if sigma_px > 0:
        # centroid noise shrinks with brightness (shot/read-noise limited)
        s = sigma_px * np.sqrt(150.0 / np.maximum(w, 1.0))
        cx = cx + rng.normal(0, 1, len(cx)) * np.minimum(s, 0.3)
        cy = cy + rng.normal(0, 1, len(cy)) * np.minimum(s, 0.3)
    det = w >= amp_min
    rows = np.zeros((len(roots), 8))
    rows[:, 1], rows[:, 2] = cx, cy
    rows[:, 3] = 9 * cnt
    rows[:, 4] = rows[:, 5] = 2 + cnt
    rows[:, 6] = np.round(w * 2 * np.pi * 0.72**2)
    new_idx = np.where(det, np.cumsum(det) - 1, -1)
    rows = rows[det]
    if extra is not None and len(extra):
        rows = np.vstack([rows, extra])
    rows[:, 7] = -1
    order = np.argsort(rows[:, 2], kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    rows = rows[order]
    rows[:, 0] = np.arange(len(rows))
    t_of = new_idx[inv]
    return rows, np.where(t_of >= 0, rank[np.maximum(t_of, 0)], -1)


def stamp_tnr(case_dir: Path) -> None:
    """Set every target's ``tnr`` to the index of the 3D point that uses it
    (-1 if none), as the image pipeline does after correspondences. The
    epipolar trackers (trackcorr, full_multipass, two_directional) read it to
    tell used from free targets; with all -1 they link nothing. Idempotent."""
    from openptv2.storage import RunStore

    store = RunStore(case_dir / "res" / "run.zarr", mode="a")
    n_out = json.loads((case_dir / "case.json").read_text())["n_out"]
    for f in range(1, n_out + 1):
        _, ids = store.read_correspondences(f)
        for cam in range(NUM_CAMS):
            t = store._targets_to_array(store.read_targets(cam, f))
            t[:, 7] = -1
            col = ids[:, cam]
            used = col >= 0
            t[col[used], 7] = np.flatnonzero(used)
            store.write_targets(cam, f, t)


def _false_targets(k: int, n_out: int) -> dict:
    """Static-scene false targets of the real image pipeline: L2 detections
    farther than 1 px from every true particle, per (output frame, camera)."""
    from openptv2.storage import RunStore
    from openptv2.storage.run_store import _frame_key

    z = np.load(SRC / "truth" / "truth.npz")
    s = RunStore.open(SRC_RUN, mode="r")
    out = {}
    for fo in range(1, n_out + 1):
        fs = (fo - 1) * k % 400 + 1
        m = z["frame"] == fs
        for cam in range(NUM_CAMS):
            rows = np.asarray(s.root[f"targets/cam_{cam}/{_frame_key(fs)}"])
            q = z["pixel_xy_per_cam"][m][:, 2 * cam: 2 * cam + 2]
            q = q[~np.isnan(q[:, 0])]
            d, _ = cKDTree(q).query(rows[:, 1:3])
            out[fo, cam] = rows[d > 1.0]
    return out


#: Measured on the real L2 pipeline of this case (frames 20..380): 3D depth
#: noise 0.094 mm (x/y 0.019), depth bias -0.17 mm from the render-
#: calibration error; 2D centroid robust sd 0.04 px; recall of isolated
#: particles 0.845 -> log-normal brightness cut at 49 grey levels.
NOISE_MM = {"L0": 0.0, "L1": 0.094, "L2": 0.094}


def build_case(level: str, mult: float, cluster: float, k: int, n_out: int,
               seed: int = 0, sigma_px: float = DEFAULT_SIGMA_PX,
               amp_min: float = 49.0) -> Path:
    from openptv2.storage import RunStore

    name = case_name(level, mult, cluster, k, n_out, sigma_px)
    case_dir = WORK / "cases" / name
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)
    _copy_cal(case_dir)
    t0 = time.perf_counter()

    if level == "L2":
        assert mult == 1 and cluster == 0, "L2 exists at native density only"
        src = np.load(SRC / "truth" / "truth.npz")
        keep = (src["frame"] - 1) % k == 0
        keep &= (src["frame"] - 1) // k < n_out
        truth = {key: src[key][keep] for key in ("particle_id", "pos_mm")}
        truth["frame"] = 1 + (src["frame"][keep] - 1) // k
        truth["blob"] = np.zeros(keep.sum(), bool)
    else:
        flow = Flow()
        truth = _advect_truth(flow, int(round(1500 * mult)), n_out, k, cluster, seed)

    bounds = _kinematic_bounds(truth, k, NOISE_MM[level])
    yaml_path = _write_yaml(case_dir, n_out, bounds)
    store = RunStore(case_dir / "res" / "run.zarr", mode="a")
    stats = {"level": level, "mult": mult, "cluster": cluster, "k": k,
             "n_out": n_out, "bounds": bounds}

    if level == "L2":
        src_store = RunStore.open(SRC_RUN, mode="r")
        for fo in range(1, n_out + 1):
            fs = 1 + (fo - 1) * k
            for cam in range(NUM_CAMS):
                store.write_targets(cam, fo, src_store.read_targets(cam, fs))
            pos, ids = src_store.read_correspondences(fs)
            store.write_correspondences(fo, pos, ids)
    else:
        exp = _experiment(case_dir, yaml_path, 1, n_out)
        cpar, cals = exp.cpar, list(exp.cals)
        render = list(_experiment(case_dir, yaml_path, 1, n_out).cals)
        rng = np.random.default_rng(seed + 2)
        if level == "L1":
            _render_cals(render, cpar)
        fr, pos, amp = truth["frame"], truth["pos_mm"], truth["amp"]
        imx, imy = cpar.imx, cpar.imy
        false_t = _false_targets(k, n_out) if level == "L1" else {}
        lost = []
        for fo in range(1, n_out + 1):
            sel = np.flatnonzero(fr == fo)
            tid = np.full((len(sel), NUM_CAMS), -1, np.int32)
            for cam in range(NUM_CAMS):
                xy = _project(pos[sel], render[cam] if level == "L1" else cals[cam], cpar)
                vis = (xy[:, 0] >= 0) & (xy[:, 0] < imx) & (xy[:, 1] >= 0) & (xy[:, 1] < imy)
                if level == "L1":
                    rows, t_of = _target_rows(xy[vis], amp[sel][vis], True, sigma_px, rng,
                                              extra=false_t[fo, cam], amp_min=amp_min)
                else:
                    rows, t_of = _target_rows(xy[vis], np.full(vis.sum(), 150.0), False, 0, rng)
                tid[np.flatnonzero(vis), cam] = t_of
                store.write_targets(cam, fo, rows)
                lost.append(1 - (t_of >= 0).sum() / max(len(sel), 1))
            if level == "L0":
                seen = (tid >= 0).sum(1) >= 2
                store.write_correspondences(fo, pos[sel][seen], tid[seen])
        stats["target_loss_frac"] = float(np.mean(lost))
        if level == "L1":
            from openptv2.correspondences import match_correspondences_batch_parallel

            match_correspondences_batch_parallel(
                frames=range(1, n_out + 1), cpar=cpar, cals=cals, vpar=exp.vpar,
                zarr_store_path=str(case_dir / "res" / "run.zarr"),
                n_workers=min(16, n_out), write_to_store=True,
            )

    np.savez_compressed(case_dir / "truth.npz", **truth)
    (case_dir / "case.json").write_text(json.dumps(stats, indent=2))
    if level != "L2":  # L2's targets already carry the pipeline's tnr
        stamp_tnr(case_dir)
    stats["build_s"] = round(time.perf_counter() - t0, 1)
    stats["particles_per_frame"] = int(np.bincount(truth["frame"]).max())
    (case_dir / "case.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))
    return case_dir


# --------------------------------------------------------------------------
# tracking
# --------------------------------------------------------------------------

_CORE_PRESETS = {
    "priority_segment_3d", "trackcorr", "standard_forward", "two_directional",
    "full_multipass", "4be",
}


def _oracle(case_dir: Path, n_out: int) -> dict:
    """Perfect linker: the case's input 3D points, linked by true identity
    (each truth particle keeps its single closest point per frame; ghosts stay
    1-point tracks). The ceiling any tracker can reach on this input."""
    from openptv2.storage import RunStore

    s = RunStore.open(case_dir, mode="r")
    tr = np.load(case_dir / "truth.npz")
    fr_all, pos_all = [], []
    for f in range(1, n_out + 1):
        p, _ = s.read_correspondences(f)
        fr_all.append(np.full(len(p), f))
        pos_all.append(p)
    frame, pos = np.concatenate(fr_all), np.vstack(pos_all)
    by_frame = {}
    for f in np.unique(tr["frame"]):
        m = tr["frame"] == f
        by_frame[int(f)] = (tr["pos_mm"][m], tr["particle_id"][m], cKDTree(tr["pos_mm"][m]))
    g0, _, off = _match(by_frame, frame, pos, 0.6)
    bias = np.median(off[g0 >= 0], axis=0) if (g0 >= 0).sum() > 100 else np.zeros(3)
    gid, _, _ = _match(by_frame, frame, pos - bias, 0.4)
    ghost = gid < 0
    trajid = np.where(ghost, gid.max() + 1 + np.arange(len(gid)), gid)
    # a real linker cannot bridge long dropouts: split identity tracks at
    # gaps longer than the repair's max_gap (3 missing frames)
    o = np.lexsort((frame, trajid))
    brk = np.r_[True, (trajid[o][1:] != trajid[o][:-1]) | (np.diff(frame[o]) > 4)]
    trajid[o] = np.cumsum(brk)
    return {"trajid": trajid, "frame": frame, "pos": pos}


def _track_optv_c(case: str, label: str, overrides: list[str]) -> Path:
    """The ORIGINAL C liboptv trackcorr (the ``optv`` package), forward only,
    on the case exported to ASCII (frames renumbered from 10001: the C
    library pads names below that). Same gates as the case YAML; ``+key=value``
    overrides apply. Reference for parity with our Cython port."""
    import re
    import tempfile

    import benchmark_utils as bu  # scripts/

    from openptv2.storage import RunStore
    from openptv2.storage.legacy import export_run

    case_dir = WORK / "cases" / case
    run_dir = WORK / "runs" / case / label
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    n_out = json.loads((case_dir / "case.json").read_text())["n_out"]
    tr = yaml.safe_load((case_dir / "parameters_Run1.yaml").read_text())["track"]
    for item in overrides:
        key, _, val = item.partition("=")
        tr[key] = yaml.safe_load(val)
    ov = {k: tr[k] for k in ("dvxmin", "dvxmax", "dvymin", "dvymax", "dvzmin", "dvzmax",
                             "dacc", "angle", "flagNewParticles") if k in tr}
    src = Path(tempfile.mkdtemp()) / "case"
    src.mkdir()
    shutil.copytree(case_dir / "cal", src / "cal")
    (src / "res").mkdir()
    (src / "img").mkdir()
    export_run(RunStore.open(case_dir, mode="r"), src)
    shutil.copy2(case_dir / "parameters_Run1.yaml", src / "parameters_Run1.yaml")
    for d in ("res", "img"):
        for f in sorted((src / d).iterdir()):
            m = re.match(r"^(.*?)\.(\d+)(_targets)?$", f.name)
            if m:
                f.rename(f.with_name(f"{m.group(1)}.{int(m.group(2)) + 10000}{m.group(3) or ''}"))
    status = "ok"
    dt = 0.0
    try:
        pred, dt = bu.run_liboptv_tracker(mode="trackcorr", track_overrides=ov, src=src,
                                          first=10001, n_frames=n_out)
        tid = np.concatenate([np.full(len(v), i) for i, v in enumerate(pred.values())])
        arr = np.array([row for v in pred.values() for row in v], float)
        np.savez_compressed(run_dir / "pred.npz", trajid=tid,
                            frame=arr[:, 0].astype(int) + 1, pos=arr[:, 1:4])
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        status = f"error: {type(e).__name__}: {e}"
    out = {"tracker": label, "case": case, "time_s": round(dt, 2), "status": status}
    (run_dir / "run.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out))
    return run_dir


def track(case: str, tracker: str) -> Path:
    """Run one tracker on an isolated copy of the case; save its trajectories.

    ``tracker`` may carry ``@fwd`` (forward pass only; default is the
    forward+backward run the case YAML asks for) and ``+key=value`` overrides
    of the ``track:`` section (YAML values), e.g. ``two_phase+leaf_weight=0``."""
    label = tracker
    tracker, *overrides = tracker.split("+")  # e.g. two_phase+confirm_tol=null
    tracker, _, mod = tracker.partition("@")
    if tracker == "optv_c":
        return _track_optv_c(case, label, overrides)
    if tracker == "oracle":
        case_dir = WORK / "cases" / case
        run_dir = WORK / "runs" / case / tracker
        run_dir.mkdir(parents=True, exist_ok=True)
        n_out = json.loads((case_dir / "case.json").read_text())["n_out"]
        t0 = time.perf_counter()
        np.savez_compressed(run_dir / "pred.npz", **_oracle(case_dir, n_out))
        out = {"tracker": tracker, "case": case,
               "time_s": round(time.perf_counter() - t0, 2), "status": "ok"}
        (run_dir / "run.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out))
        return run_dir
    from openptv2.benchmarking.runner import read_trajectories
    from openptv2.plugins import run_tracking_plugin

    case_dir = WORK / "cases" / case
    run_dir = WORK / "runs" / case / label
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    for sub in ("cal", "img"):
        shutil.copytree(case_dir / sub, run_dir / sub)
    (run_dir / "res").mkdir()
    shutil.copytree(case_dir / "res" / "run.zarr", run_dir / "res" / "run.zarr")
    yaml_path = run_dir / "parameters_Run1.yaml"
    shutil.copy2(case_dir / "parameters_Run1.yaml", yaml_path)
    raw = yaml.safe_load(yaml_path.read_text())
    n_out = int(raw["sequence"]["last"])
    if mod == "fwd":
        raw["track"].update(run_backward=False, direction="forward", bidirectional=False)
    for item in overrides:
        key, _, val = item.partition("=")
        raw["track"][key] = yaml.safe_load(val)
    if mod == "fwd" or overrides:
        yaml_path.write_text(yaml.safe_dump(raw, sort_keys=False))

    exp = _experiment(run_dir, yaml_path, 1, n_out)
    if tracker == "priority_segment_3d":
        exp.track3d = True
    if tracker in _CORE_PRESETS:
        exp.pm.parameters.setdefault("plugins", {})["selected_tracking"] = tracker
    cwd = os.getcwd()
    status = "ok"
    t0 = time.perf_counter()
    try:
        os.chdir(run_dir)
        run_tracking_plugin(tracker, exp)
    except Exception as e:  # keep the sweep going; record the failure
        import traceback

        traceback.print_exc()
        status = f"error: {type(e).__name__}: {e}"
    finally:
        os.chdir(cwd)
    if status == "ok" and raw["track"].get("postconfirm_tol") is not None:
        from openptv2.storage import RunStore
        from openptv2.tracking_postprocess import confirm_links

        t1 = time.perf_counter()
        st = confirm_links(
            "res/ptv_is", 1, n_out, float(raw["track"]["postconfirm_tol"]),
            ends=bool(raw["track"].get("postconfirm_ends", False)),
            store=RunStore(run_dir / "res" / "run.zarr", mode="a"),
        )
        print("confirm_links", st, f"{time.perf_counter() - t1:.1f}s")
    dt = time.perf_counter() - t0
    out = {"tracker": label, "case": case, "time_s": round(dt, 2), "status": status}
    if status == "ok":
        tracks = read_trajectories(run_dir / "res", 1, n_out, NUM_CAMS)
        tid = np.concatenate([np.full(len(v), i) for i, v in enumerate(tracks.values())])
        arr = np.array([row for v in tracks.values() for row in v], float)
        np.savez_compressed(run_dir / "pred.npz", trajid=tid, frame=arr[:, 0].astype(int),
                            pos=arr[:, 1:4])
    (run_dir / "run.json").write_text(json.dumps(out, indent=2))
    # the copied store is large and fully reproducible
    shutil.rmtree(run_dir / "res")
    print(json.dumps(out))
    return run_dir


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def _match(truth_by_frame, frame, pos, eps):
    """Per-point truth particle id (-1 = ghost/unmatched), one-to-one per frame,
    closest first."""
    out = np.full(len(frame), -1, np.int64)
    dist = np.full(len(frame), np.inf)
    offset = np.zeros((len(frame), 3))
    for f in np.unique(frame):
        if f not in truth_by_frame:
            continue
        tpos, tids, tree = truth_by_frame[f]
        sel = np.flatnonzero(frame == f)
        d, j = tree.query(pos[sel], distance_upper_bound=eps)
        ok = np.isfinite(d)
        sel, d, j = sel[ok], d[ok], j[ok]
        o = np.argsort(d)
        sel, d, j = sel[o], d[o], j[o]
        _, first = np.unique(j, return_index=True)
        out[sel[first]] = tids[j[first]]
        dist[sel[first]] = d[first]
        offset[sel[first]] = pos[sel[first]] - tpos[j[first]]
    return out, dist, offset


def _smooth(trajid, frame, pos, fps, window=9, order=3, min_window=5):
    from flowtracks.smoothing import savitzky_golay
    from flowtracks.trajectory import Trajectory

    o = np.lexsort((frame, trajid))
    trajid, frame, pos = trajid[o], frame[o], pos[o]
    cuts = np.flatnonzero(np.diff(trajid)) + 1
    trajs = [
        Trajectory(p, np.zeros_like(p), f, int(t[0]))
        for t, f, p in zip(np.split(trajid, cuts), np.split(frame, cuts), np.split(pos, cuts))
    ]
    sm = savitzky_golay(trajs, fps, window, order, min_window=min_window)
    if not sm:
        z = np.zeros((0, 3))
        return np.zeros(0, int), np.zeros(0, int), z, z, z
    return (
        np.concatenate([np.full(len(t.time()), t.trajid()) for t in sm]),
        np.concatenate([np.asarray(t.time()) for t in sm]),
        np.concatenate([t.pos() for t in sm]),
        np.concatenate([t.velocity() for t in sm]),
        np.concatenate([t.accel() for t in sm]),
    )


def evaluate_pred(case_dir: Path, pred: dict, flow: Flow, eps: float = 0.4,
                  max_gap: int = 3, windows=(9, 21, 41, 81), min_len: int = 5) -> dict:
    from flowtracks.repair import repair_arrays

    info = json.loads((case_dir / "case.json").read_text())
    k = int(info["k"])
    fps = FPS / k
    tr = np.load(case_dir / "truth.npz")
    t_id, t_fr, t_pos = tr["particle_id"], tr["frame"], tr["pos_mm"]
    by_frame = {}
    for f in np.unique(t_fr):
        m = t_fr == f
        by_frame[int(f)] = (t_pos[m], t_id[m], cKDTree(t_pos[m]))

    trajid, frame, pos = pred["trajid"], pred["frame"], pred["pos"]
    r: dict = {}

    # global calibration bias (the render calibration differs from the
    # processing one): not the tracker's doing, removed before matching
    g0, _, off = _match(by_frame, frame, pos, 0.6)
    bias = np.median(off[g0 >= 0], axis=0) if (g0 >= 0).sum() > 100 else np.zeros(3)
    r["bias_mm"] = [round(float(b), 4) for b in bias]
    pos = pos - bias

    # ---- raw tracker output: link correctness --------------------------
    gid, _, _ = _match(by_frame, frame, pos, eps)
    o = np.lexsort((frame, trajid))
    a, b = o[:-1], o[1:]
    link = trajid[a] == trajid[b]
    a, b = a[link], b[link]
    good = (gid[a] >= 0) & (gid[a] == gid[b])
    r["raw_links"] = int(len(a))
    r["raw_link_precision"] = float(good.mean()) if len(a) else 0.0
    # true links whose both ends exist as input 3D points = what a tracker
    # could possibly find; recall against those isolates the tracker
    t_o = np.lexsort((t_fr, t_id))
    ta, tb = t_o[:-1], t_o[1:]
    tl = (t_id[ta] == t_id[tb]) & (t_fr[tb] == t_fr[ta] + 1)
    n_true_links = int(tl.sum())
    # keys (id, frame) of reconstructed (matched) truth points
    key = lambda i, f: i.astype(np.int64) * 100000 + f  # noqa: E731
    have = np.unique(key(gid[gid >= 0], frame[gid >= 0]))
    ta, tb = ta[tl], tb[tl]
    reach = np.isin(key(t_id[ta], t_fr[ta]), have) & np.isin(key(t_id[tb], t_fr[tb]), have)
    step1 = good & (frame[b] == frame[a] + 1)
    r["raw_gap_links"] = int((good & ~step1).sum())
    found = np.unique(key(gid[a][step1], frame[a][step1]))
    r["raw_link_recall"] = float(len(found) / max(n_true_links, 1))
    r["raw_link_recall_reachable"] = float(len(found) / max(int(reach.sum()), 1))
    r["input_point_recall"] = float(len(have) / len(t_id))

    # ---- postptv repair + smoothing ------------------------------------
    t0 = time.perf_counter()
    new_id, rep = repair_arrays(trajid, frame, pos, max_gap=max_gap)
    r["postptv_s"] = round(time.perf_counter() - t0, 2)
    r["repair"] = {kk: rep.get(kk) for kk in ("links_cut", "points_attached", "joins", "threshold")}

    # ---- final output vs the true flow ---------------------------------
    # Smoothing window matters more than anything at this noise level, so
    # each tracker is scored at its own best window (a user would tune it);
    # the true field is evaluated at the smoothed (bias-corrected) positions,
    # so ghost points count against the flow like everything else.
    r["by_window"] = {}
    for w in windows:
        if pred.get("w") is not None:
            # per-point quality weights (A10 after tracking): weighted
            # Savitzky-Golay; unit weights reproduce the flowtracks filter
            from openptv2.quality_post import weighted_savgol

            s_id, s_fr, s_pos, s_vel, s_acc = weighted_savgol(
                new_id, frame, pos, pred["w"], fps, w, 3, min_len
            )
        else:
            s_id, s_fr, s_pos, s_vel, s_acc = _smooth(
                new_id, frame, pos, fps, w, 3, min_len
            )
        n_out = len(s_fr)
        if not n_out:
            continue
        u_true = np.zeros_like(s_vel)
        a_true = np.zeros_like(s_acc)
        for f in np.unique(s_fr):
            m = s_fr == f
            tn = 1.0 + (f - 1) * k
            u_true[m] = flow.velocity(s_pos[m], tn) * FPS  # mm/frame -> mm/s
            a_true[m] = flow.material_acc(s_pos[m], tn) * FPS**2
        ev = np.linalg.norm(s_vel - u_true, axis=1)
        ea = np.linalg.norm(s_acc - a_true, axis=1)
        u_rms = float(np.sqrt(np.mean(np.sum(u_true**2, 1))))
        a_rms = float(np.sqrt(np.mean(np.sum(a_true**2, 1))))
        r["by_window"][w] = {
            "vel": float(np.sqrt(np.mean(ev**2)) / u_rms),
            "vel_p95": float(np.percentile(ev, 95) / u_rms),
            "acc": float(np.sqrt(np.mean(ea**2)) / a_rms),
            "u_rms_m_s": u_rms / 1000, "a_rms_m_s2": a_rms / 1000,
        }
        if w == windows[0]:
            keep = (s_id, s_fr, s_pos)
    bw = r["by_window"]
    if bw:
        wv = min(bw, key=lambda w: bw[w]["vel"])
        wa = min(bw, key=lambda w: bw[w]["acc"])
        r.update(vel_err_rel=bw[wv]["vel"], vel_err_rel_p95=bw[wv]["vel_p95"], vel_win=wv,
                 acc_err_rel=bw[wa]["acc"], acc_win=wa)
    else:
        keep = (np.zeros(0, int), np.zeros(0, int), np.zeros((0, 3)))
        r.update(vel_err_rel=1.0, vel_err_rel_p95=1.0, vel_win=0, acc_err_rel=1.0, acc_win=0)
    s_id, s_fr, s_pos = keep
    g2, d2, _ = _match(by_frame, s_fr, s_pos, eps)
    n_out = len(s_fr)
    r["out_points"] = int(n_out)
    r["ghost_point_frac"] = float((g2 < 0).mean()) if n_out else 1.0
    matched = g2 >= 0
    r["pos_err_mm"] = float(np.sqrt(np.mean(d2[matched] ** 2))) if matched.any() else 0.0

    # trajectories: purity and long-track completeness
    if n_out:
        o = np.lexsort((g2, s_id))
        sid, gg = s_id[o], g2[o]
        pair, cnt = np.unique(np.c_[sid, gg], axis=0, return_counts=True)
        traj_len = np.bincount(np.unique(sid, return_inverse=True)[1])
        # majority id per trajectory
        ids_u = np.unique(sid)
        best = np.zeros(len(ids_u), int)
        best_id = np.full(len(ids_u), -1)
        pos_in = np.searchsorted(ids_u, pair[:, 0])
        for (t, g), c, pi in zip(pair, cnt, pos_in):
            if g >= 0 and c > best[pi]:
                best[pi], best_id[pi] = c, g
        r["n_traj"] = int(len(ids_u))
        r["traj_purity"] = float(best.sum() / max(traj_len.sum(), 1))
        r["pure_traj_frac"] = float(np.mean(best / traj_len >= 0.95))
        r["ghost_traj_frac"] = float(np.mean(best / traj_len < 0.5))
        r["mean_traj_len"] = float(traj_len.mean())
        # per truth id: longest single output trajectory following it
        cover = {}
        for (t, g), c in zip(pair, cnt):
            if g >= 0 and c > cover.get(g, 0):
                cover[g] = c
        pieces = {}
        for (t, g), c in zip(pair, cnt):
            if g >= 0 and c >= min_len:
                pieces[g] = pieces.get(g, 0) + 1
    else:
        cover, pieces = {}, {}
        r.update(n_traj=0, traj_purity=0.0, pure_traj_frac=0.0, ghost_traj_frac=1.0,
                 mean_traj_len=0.0)
    tlen = np.bincount(t_id)
    long_ids = np.flatnonzero(tlen >= 50)
    comp = np.array([cover.get(i, 0) / tlen[i] for i in long_ids])
    r["n_true_long"] = int(len(long_ids))
    r["long_completeness"] = float(comp.mean()) if len(comp) else 0.0
    r["long_whole_frac"] = float(np.mean(comp >= 0.9)) if len(comp) else 0.0
    r["long_fragments"] = float(np.mean([pieces.get(i, 0) for i in long_ids])) if len(long_ids) else 0.0
    r["yield"] = float(matched.sum() / len(t_id))
    return r


def evaluate(case: str, trackers: list[str] | None = None) -> list[dict]:
    case_dir = WORK / "cases" / case
    flow = Flow()
    rows = []
    for run_dir in sorted((WORK / "runs" / case).iterdir()):
        if trackers and run_dir.name not in trackers:
            continue
        if not (run_dir / "run.json").exists():
            rows.append({"case": case, "tracker": run_dir.name, "status": "no result (crashed?)"})
            continue
        meta = json.loads((run_dir / "run.json").read_text())
        row = {"case": case, "tracker": run_dir.name, "time_s": meta["time_s"],
               "status": meta["status"]}
        if (run_dir / "pred.npz").exists():
            pred = dict(np.load(run_dir / "pred.npz"))
            row.update(evaluate_pred(case_dir, pred, flow))
        (run_dir / "eval.json").write_text(json.dumps(row, indent=2))
        rows.append(row)
    _print_table(rows)
    return rows


def _print_table(rows: list[dict]) -> None:
    cols = [
        ("tracker", "{:<20}", 20), ("time_s", "{:>7.1f}", 7),
        ("raw_link_precision", "{:>7.4f}", 7), ("raw_link_recall_reachable", "{:>7.4f}", 7),
        ("vel_err_rel", "{:>7.4f}", 7), ("acc_err_rel", "{:>7.3f}", 7),
        ("ghost_point_frac", "{:>7.4f}", 7), ("long_completeness", "{:>7.3f}", 7),
        ("long_fragments", "{:>6.2f}", 6), ("yield", "{:>6.3f}", 6),
    ]
    hdr = ["tracker", "time", "lnkP", "lnkR*", "vErr", "aErr", "ghost", "longC", "frag", "yield"]
    print(" ".join(f"{h:>{w}}" if i else f"{h:<{w}}" for i, (h, (_, _, w)) in enumerate(zip(hdr, cols))))
    for r in sorted(rows, key=lambda r: r.get("vel_err_rel", 9)):
        if "vel_err_rel" not in r:
            print(f"{r['tracker']:<20} {r['status'][:90]}")
            continue
        print(" ".join(fmt.format(r[c]) for c, fmt, _ in cols))


def report() -> None:
    """Pivot every stored eval.json: one table per metric, trackers x cases;
    written to WORK/results.md and results.csv."""
    import csv

    rows = [json.loads(p.read_text()) for p in sorted((WORK / "runs").glob("*/*/eval.json"))]
    rows = [r for r in rows if "vel_err_rel" in r]
    cases = sorted({r["case"] for r in rows})
    trackers = sorted({r["tracker"] for r in rows})
    metrics = [
        ("vel_err_rel", "velocity error / rms speed (best window) - lower is better", "{:.3f}"),
        ("acc_err_rel", "acceleration error / rms accel (best window)", "{:.2f}"),
        ("ghost_point_frac", "ghost fraction of output points", "{:.3f}"),
        ("ghost_traj_frac", "fraction of trajectories that follow no true particle", "{:.3f}"),
        ("long_completeness", "true 50+ frame trajectories: mean fraction kept in one piece", "{:.3f}"),
        ("long_fragments", "pieces per true 50+ frame trajectory (1 = never broken)", "{:.2f}"),
        ("raw_link_precision", "raw tracker link precision (before postptv)", "{:.3f}"),
        ("raw_link_recall_reachable", "raw link recall over links whose ends both exist", "{:.3f}"),
        ("yield", "true particle positions recovered in output", "{:.3f}"),
        ("time_s", "tracker wall time, s (concurrent runs: relative only)", "{:.0f}"),
    ]
    by = {(r["tracker"], r["case"]): r for r in rows}
    out = ["# Tracker benchmark\n"]
    for key, title, fmt in metrics:
        out.append(f"\n## {title}\n")
        out.append("| tracker | " + " | ".join(cases) + " |")
        out.append("|---|" + "---:|" * len(cases))
        for t in trackers:
            cells = [fmt.format(by[t, c][key]) if (t, c) in by and key in by[t, c] else "" for c in cases]
            out.append(f"| {t} | " + " | ".join(cells) + " |")
    (WORK / "results.md").write_text("\n".join(out) + "\n")
    keys = [m[0] for m in metrics] + ["vel_win", "acc_win", "n_traj", "mean_traj_len", "pos_err_mm"]
    with open(WORK / "results.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "tracker"] + keys)
        for r in rows:
            w.writerow([r["case"], r["tracker"]] + [r.get(k, "") for k in keys])
    print("\n".join(out))


# --------------------------------------------------------------------------
# sweep driver: one process per (case, tracker)
# --------------------------------------------------------------------------


def sweep(cases: list[str], trackers: list[str], jobs: int, timeout: float) -> None:
    todo = [(c, t) for c in cases for t in trackers]
    running: list[tuple[subprocess.Popen, str, str, float]] = []
    log_dir = WORK / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    while todo or running:
        while todo and len(running) < jobs:
            c, t = todo.pop(0)
            log = open(log_dir / f"{c}__{t}.log", "w")
            p = subprocess.Popen(
                [sys.executable, __file__, "track", "--case", c, "--tracker", t],
                stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "OMP_NUM_THREADS": "1"},
            )
            running.append((p, c, t, time.time()))
        time.sleep(2)
        for item in list(running):
            p, c, t, t0 = item
            if p.poll() is not None:
                running.remove(item)
                print(f"done {c} {t} rc={p.returncode} {time.time() - t0:.0f}s", flush=True)
            elif time.time() - t0 > timeout:
                p.kill()
                running.remove(item)
                rd = WORK / "runs" / c / t
                rd.mkdir(parents=True, exist_ok=True)
                (rd / "run.json").write_text(json.dumps(
                    {"tracker": t, "case": c, "time_s": timeout, "status": f"timeout {timeout:.0f}s"}))
                print(f"TIMEOUT {c} {t}", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--level", choices=["L0", "L1", "L2"], default="L1")
    b.add_argument("--mult", type=float, default=1.0)
    b.add_argument("--cluster", type=float, default=0.0)
    b.add_argument("--k", type=int, default=1)
    b.add_argument("--n-out", type=int, default=200)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--sigma-px", type=float, default=DEFAULT_SIGMA_PX,
                   help="blob-centre noise (px); ~0.06 matches real jitter")
    t = sub.add_parser("track")
    t.add_argument("--case", required=True)
    t.add_argument("--tracker", required=True)
    sub.add_parser("report", help="pivot all stored evaluations")
    st = sub.add_parser("stamp", help="(re)write tnr into a case's targets")
    st.add_argument("--cases", nargs="+", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--case", required=True)
    e.add_argument("--trackers", nargs="*")
    s = sub.add_parser("sweep")
    s.add_argument("--cases", nargs="+", required=True)
    s.add_argument("--trackers", nargs="*", default=TRACKERS)
    s.add_argument("-j", "--jobs", type=int, default=6)
    s.add_argument("--timeout", type=float, default=3600)
    s.add_argument("--no-eval", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        build_case(args.level, args.mult, args.cluster, args.k, args.n_out, args.seed,
                   args.sigma_px)
    elif args.cmd == "track":
        track(args.case, args.tracker)
    elif args.cmd == "report":
        report()
    elif args.cmd == "stamp":
        for c in args.cases:
            stamp_tnr(WORK / "cases" / c)
            print("stamped", c)
    elif args.cmd == "eval":
        evaluate(args.case, args.trackers)
    elif args.cmd == "sweep":
        sweep(args.cases, args.trackers, args.jobs, args.timeout)
        if not args.no_eval:
            for c in args.cases:
                print(f"\n=== {c}")
                evaluate(c)


if __name__ == "__main__":
    main()
