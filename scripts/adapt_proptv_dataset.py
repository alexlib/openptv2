"""Adapt a proPTV origin_*.txt case into an openptv2 tracker-benchmark dataset.

proPTV's ground truth already matches openptv2's own origin_*.txt convention
(``ID,X,Y,Z,...`` -- see scripts/benchmark_utils.read_gt_frames, which already
calls this format "proPTV-style"). This does NOT re-run proPTV's own
detection/correspondence/calibration: 500_30 has no saved reconstruction at
all, and 500_25's own triangulation is ~exact anyway (reconstruction error
~1e-7, see docs/plans/2026-08-17-lagrangian-accuracy-program.md, Phase 2's
proPTV note) so there is nothing to gain from re-deriving it. Instead it
feeds proPTV's true positions straight in as each frame's point cloud --
exactly what our own synthetic sets already do at their own near-zero-noise
level -- so Phase 1b's injectable-noise knob is the intended way to make this
realistic, applied uniformly to both datasets.

proPTV has no "real" camera system to match -- its xc0,yc0,...,xc3,yc3
columns are just another simulator's own private, unrelated camera model, and
matching it (2026-08-17's DLT-fit attempt in calibrate_proptv_dlt.py) turned
out to be pure downside: no simpler than defining our own cameras, and it
surfaced a genuine sign bug in the shared ray_tracing Snell's-law code (fixed
2026-08-18, see the plan doc) whenever a camera looks the "other way" through
the glass-normal convention -- exactly what a DLT-fit rig can produce by
accident. So this script ignores proPTV's own pixel columns entirely and
instead: (1) reuses the scaffold dataset's own cal/ as-is (already a working,
tested 4-camera rig -- no calibration step needed at all), (2) rescales
proPTV's [0,1]-cube XYZ into that rig's working volume (affine, cube-center ->
origin, extent -> +-20mm, safely inside the scaffold's own +-27..58mm span),
and (3) generates each camera's 2D targets itself via img_coord() on the
rescaled truth -- self-consistent by construction, zero calibration-matching
residual, and per-camera visibility (in vs. out of frame) falls out of the
same projection instead of trusting proPTV's own visibility flags.

Bonus: because positions are now mm-scale like every other openptv2 dataset,
eps/distance-tolerance metric arguments (default eps=1.0 in
scripts/benchmark_utils.combined_metrics) no longer need special-casing for
this dataset the way the old [0,1]-cube convention did.

Both per-camera 2D targets AND 3D correspondences go through the unified
RunStore (`res/run.zarr`, `RunStore.write_targets` / `write_correspondences`)
-- NOT ascii `_targets`/`rt_is` files, no legacy `.par` files anywhere, and no
tracker-specific special-casing: `py_trackcorr_init(exp)` is the single
factory every tracker (priority_segment_3d, trackcorr, all the rest) uses to
build its `Tracker`, and it attaches this same store; `read_path_frame`
(`tracking_frame_buf.py`) checks `store.has_correspondences(frame)` before
ever touching ascii. So every tracker reads the identical data, from the
identical store, populated the identical way here -- not "trackcorr gets 2D
targets, priority_segment_3d gets rt_is": both get correspondences from the
store, trackcorr additionally reads the store's 2D targets for its epipolar
search. Camera-index columns in the correspondences (previously always -1 in
the ascii convention) now hold each particle's actual 0-based position within
that camera/frame's target array in the store, i.e. exactly what a real
correspondence stage would have produced.
"""

from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path

import numpy as np
import yaml

FIRST = 10001
NUM_CAMS = 4
CUBE_SCALE = 40.0  # [0,1]^3 -> [-20,20]^3, inside the scaffold rig's FOV


def _prepare_scaffold(scaffold: Path, out: Path):
    """Clone the scaffold, wipe its own res/img output, and return
    (yaml_path, yaml_data, ptv, cpar) with the multimedia model overridden
    to plain air -- proPTV has no glass/water, this is a pure pinhole model.
    Shared by convert() and convert_realistic()."""
    from openptv2.algorithms.parameters import ControlPar, MmNp

    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(scaffold, out)
    res, img = out / "res", out / "img"
    # Scaffold's own res/ and img/ are that dataset's leftover ascii/zarr
    # output (added.*, ptv_is.*, rt_is.*, run.zarr, camN.<frame>_targets,
    # ...) -- none of it applies to this dataset; wipe both clean rather
    # than pick individual globs.
    shutil.rmtree(res)
    res.mkdir()
    shutil.rmtree(img)
    img.mkdir()

    yaml_path = out / "parameters_Run1.yaml"
    yaml_data = yaml.safe_load(yaml_path.read_text())
    ptv = yaml_data["ptv"]
    ptv["mmp_n1"] = 1.0
    ptv["mmp_n2"] = 1.0
    ptv["mmp_n3"] = 1.0
    ptv["mmp_d"] = 0.0
    cpar = ControlPar(
        num_cams=NUM_CAMS,
        imx=ptv["imx"],
        imy=ptv["imy"],
        pix_x=ptv["pix_x"],
        pix_y=ptv["pix_y"],
        mm=MmNp(n1=1.0, n2=[1.0], n3=1.0, d=[0.0]),
    )
    return yaml_path, yaml_data, ptv, cpar


def convert(proptv_case_dir: Path, scaffold: Path, out: Path) -> None:
    from openptv2.algorithms.calibration import Calibration
    from openptv2.algorithms.imgcoord import img_coord
    from openptv2.algorithms.tracking_frame_buf import Target
    from openptv2.algorithms.trafo import metric_to_pixel
    from openptv2.storage import RunStore

    origin_dir = proptv_case_dir / "origin"
    files = sorted(origin_dir.glob("origin_*.txt"))
    if not files:
        raise FileNotFoundError(f"no origin_*.txt under {origin_dir}")

    yaml_path, yaml_data, ptv, cpar = _prepare_scaffold(scaffold, out)
    res = out / "res"
    cals = [
        Calibration.from_file(
            str(out / "cal" / f"cam{c + 1}.tif.ori"), str(out / "cal" / f"cam{c + 1}.tif.addpar")
        )
        for c in range(NUM_CAMS)
    ]

    store = RunStore.open(out, mode="a")

    for i, f in enumerate(files):
        fn = FIRST + i
        rows = []
        for line in f.read_text().strip().splitlines():
            if line.startswith("#"):
                continue
            parts = [float(p) for p in line.split()]
            pid = int(parts[0])
            # Rescale proPTV's [0,1]^3 cube into the scaffold rig's working
            # volume; ignore proPTV's own xc/yc columns (parts[9:]) entirely.
            x = (parts[1] - 0.5) * CUBE_SCALE
            y = (parts[2] - 0.5) * CUBE_SCALE
            z = (parts[3] - 0.5) * CUBE_SCALE
            pix = []
            for c in range(NUM_CAMS):
                mx, my = img_coord((x, y, z), cals[c], cpar.mm)
                px, py = metric_to_pixel(mx, my, cpar)
                if 0 <= px <= cpar.imx and 0 <= py <= cpar.imy:
                    pix.append((px, py))
                else:
                    pix.append((float("nan"), float("nan")))
            rows.append((pid, x, y, z, pix))

        with open(res / f"origin_{fn}.txt", "w") as out_f:
            out_f.write("ID,X,Y,Z\n")
            for pid, x, y, z, _pix in rows:
                out_f.write(f"{pid},{x:.6f},{y:.6f},{z:.6f}\n")

        # Build each camera's target list for this frame (only particles it
        # saw), sorted by y-pixel -- REQUIRED, not cosmetic: the real
        # candidate search (candsearch_in_pix_fast_nogil,
        # track_kernels_search.py) does a binary-search jump into the
        # target array assuming targ_y is sorted, then linear-scans with an
        # early `break` the moment it sees y > ymax -- both silently wrong
        # on unsorted input (the break in particular can terminate the scan
        # before ever reaching a true candidate). `gui/ptv.py` always calls
        # `targs.sort_y()` before targets reach this code path; found
        # 2026-08-17 as the likely root cause of trackcorr's near-total
        # candidate-search failure on this adapted data (see docs/plans/
        # 2026-08-17-lagrangian-accuracy-program.md, next-steps item 2).
        cam_ids_per_row = [[-1] * NUM_CAMS for _ in rows]
        for c in range(NUM_CAMS):
            raw = []  # (row_idx, xc, yc)
            for row_idx, (_pid, _x, _y, _z, pix) in enumerate(rows):
                xc, yc = pix[c]
                if xc != xc or yc != yc:  # NaN check without importing numpy/math
                    continue
                raw.append((row_idx, xc, yc))
            raw.sort(key=lambda t: t[2])  # sort by y-pixel

            targets = []
            for pos, (row_idx, xc, yc) in enumerate(raw):
                cam_ids_per_row[row_idx][c] = pos
                # tnr is the reverse link back to this particle's row in the
                # frame's correspondence array (path_x_2 etc). Left unset
                # (defaults to 0), EVERY candidate the search finds resolves
                # to particle 0 regardless of which target actually matched
                # -- found 2026-08-18 tracing trackcorr's "always links
                # exactly 1 particle, always index 0" behaviour back to this.
                targets.append(Target(pnr=row_idx, x=xc, y=yc, tnr=row_idx))
            store.write_targets(c, fn, targets)

        # Correspondences go through the store too (RunStore.write_correspondences),
        # NOT ascii rt_is -- read_path_frame checks store.has_correspondences()
        # first and both priority_segment_3d and trackcorr build their Tracker
        # via the same py_trackcorr_init(exp) factory (which attaches this same
        # store), so this is the single shared data source for every tracker,
        # not a format some trackers see and others don't.
        pos_3d = np.array([[x, y, z] for _pid, x, y, z, _pix in rows], dtype=np.float64)
        cam_ids = np.array([cam_ids_per_row[idx] for idx in range(len(rows))], dtype=np.int32)
        store.write_correspondences(frame=fn, pos_3d=pos_3d, cam_target_ids=cam_ids)

    last = FIRST + len(files) - 1
    yaml_data["sequence"]["first"] = FIRST
    yaml_data["sequence"]["last"] = last
    yaml_path.write_text(yaml.safe_dump(yaml_data, sort_keys=False))

    print(f"wrote {len(files)} frames ({FIRST}-{last}) -> {out} (targets in res/run.zarr)")


def _perturb_calibration(cal, rng: np.random.Generator, angle_sigma_deg: float,
                          pos_sigma_mm: float, cc_ppm: float):
    """A copy of `cal` with a small, fixed-per-camera offset in position,
    orientation, and focal length -- simulating real calibration residual
    (the reconstruction-side model never matches the true camera exactly).
    Systematic per camera (not re-drawn per frame), since a real calibration
    is wrong the same way for the whole run, not randomly frame to frame."""
    out_cal = copy.deepcopy(cal)
    ext = out_cal.ext_par
    ext.x0 += rng.normal(0, pos_sigma_mm)
    ext.y0 += rng.normal(0, pos_sigma_mm)
    ext.z0 += rng.normal(0, pos_sigma_mm)
    sigma_rad = np.deg2rad(angle_sigma_deg)
    ext.omega += rng.normal(0, sigma_rad)
    ext.phi += rng.normal(0, sigma_rad)
    ext.kappa += rng.normal(0, sigma_rad)
    ext.compute_rotation_matrix()
    out_cal.int_par.cc *= 1.0 + rng.normal(0, cc_ppm * 1e-6)
    return out_cal


#: Three calibrated operating points (see convert_realistic's docstring for
#: what each knob does). "moderate" is 2026-08-18's original default;
#: severity scales noise/dropout/merge/calibration together rather than
#: leaving them at one arbitrary combination, and eps0 is DERIVED (below),
#: not listed here -- it must track noise_px, not be picked independently.
SEVERITY_PRESETS: dict[str, dict[str, float]] = {
    "mild": dict(noise_px=0.08, dropout_p=0.01, merge_radius_px=1.0,
                 calib_angle_sigma_deg=0.01, calib_pos_sigma_mm=0.01, calib_cc_ppm=100.0),
    "moderate": dict(noise_px=0.15, dropout_p=0.03, merge_radius_px=2.0,
                      calib_angle_sigma_deg=0.02, calib_pos_sigma_mm=0.02, calib_cc_ppm=200.0),
    "severe": dict(noise_px=0.3, dropout_p=0.06, merge_radius_px=3.0,
                    calib_angle_sigma_deg=0.04, calib_pos_sigma_mm=0.04, calib_cc_ppm=400.0),
}


def _derive_eps0_mm(noise_px: float) -> float:
    """eps0 (epipolar-band tolerance for accepting a correspondence) must
    track the actual detection-noise level, not sit at one fixed value --
    confirmed empirically: 0.1mm (the scaffold's own, tuned for a different
    dataset's density) produces real geometric ghost matches at this
    dataset's ~5mm particle spacing even with ZERO added noise; 0.01mm was
    clean at zero noise. Floor covers residual/rounding slack even at
    noise_px=0; the linear term is calibrated so noise_px=0.15 (the
    original "moderate" default) lands at ~0.03mm, the value already
    verified empirically to give a plausible (not ~0%, not ~100%) match
    rate. Calibration-residual severity is deliberately NOT folded in here
    -- it's a separate, independently observable contamination axis, not
    something eps0 should quietly absorb."""
    return 0.01 + 0.13 * noise_px


def _streak_dropout_mask(
    n_items: int, n_frames: int, dropout_p: float, mean_streak_frames: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """(n_items, n_frames) bool array, True = dropped that frame. A 2-state
    per-item Markov chain (visible/occluded), steady-state occlusion
    probability `dropout_p`, mean occluded-streak length
    `mean_streak_frames` -- real occlusion/defocus events last several
    consecutive frames, not independent per-frame coin flips. IID dropout
    compounds unrealistically over a long sequence (independent p=0.03/cam
    over 30 frames left almost no trajectory intact by frame ~15); a real
    particle that goes out of focus tends to stay that way for a few
    frames, then come back -- same long-run miss rate, very different
    trajectory-length distribution."""
    if dropout_p <= 0 or n_frames == 0:
        return np.zeros((n_items, n_frames), dtype=bool)
    p_leave = 1.0 / max(mean_streak_frames, 1.0)
    p_enter = dropout_p * p_leave / max(1.0 - dropout_p, 1e-9)
    state = rng.random(n_items) < dropout_p  # start in steady state
    mask = np.zeros((n_items, n_frames), dtype=bool)
    for fi in range(n_frames):
        mask[:, fi] = state
        r = rng.random(n_items)
        state = np.where(state, r >= p_leave, r < p_enter)
    return mask


def convert_realistic(
    proptv_case_dir: Path,
    scaffold: Path,
    out: Path,
    noise_px: float = 0.15,
    dropout_p: float = 0.03,
    mean_streak_frames: float = 4.0,
    merge_radius_px: float = 2.0,
    calib_angle_sigma_deg: float = 0.02,
    calib_pos_sigma_mm: float = 0.02,
    calib_cc_ppm: float = 200.0,
    eps0_mm: float | None = None,
    seed: int = 0,
) -> None:
    """Same ground truth as convert(), but runs the actual detection ->
    correspondence -> triangulation pipeline instead of injecting ground
    truth directly, simulating the real error chain (see docs/plans/
    2026-08-17-lagrangian-accuracy-program.md):

    1. Detection/centroid noise: Gaussian pixel noise on every projection
       (default 0.15px, typical for a well-focused isolated real particle).
    2. Particle-image merging: two noisy detections in the same camera
       within `merge_radius_px` collapse to one (segmentation can't tell
       them apart) -- drops one, corrupting or losing that correspondence.
    3. Missed detections: each (particle, camera) independently follows a
       2-state occlusion Markov chain with steady-state miss probability
       `dropout_p` and mean streak length `mean_streak_frames`, NOT an
       independent per-frame coin flip -- see _streak_dropout_mask.
    4. Real correspondence solving: openptv2's actual multi-camera epipolar
       matcher (openptv2.algorithms.correspondences.correspondences, the
       same one gui/ptv.py's real sequence loop uses on real experiments)
       runs on the noisy/merged/dropped 2D targets -- this is what can
       produce ghost 3D points (wrong ray intersections that coincidentally
       satisfy the criteria.eps0 tolerance) and miss real ones, not a
       simulated approximation of it. eps0_mm defaults to _derive_eps0_mm
       (tracks noise_px) rather than one fixed constant.
    5. Calibration residual: correspondence-solving and triangulation use a
       per-camera-perturbed calibration (_perturb_calibration), never the
       exact one used to generate the true projections -- systematic, not
       IID, matching how a real calibration is wrong the same way for the
       whole run.

    Target .tnr (see convert()'s docstring on the tnr bug) comes for free
    here: openptv2.algorithms.correspondences.correspondences sets it as a
    side effect of solving real correspondences (frm.targets[cam][idx].tnr
    = i), so writing targets from `frm.targets` after solving is correct by
    construction, not something this script has to get right by hand.
    """
    if eps0_mm is None:
        eps0_mm = _derive_eps0_mm(noise_px)
    from openptv2.algorithms.calibration import Calibration
    from openptv2.algorithms.correspondences import (
        correspondences as alg_correspondences,
    )
    from openptv2.algorithms.imgcoord import img_coord
    from openptv2.algorithms.orientation import point_positions as alg_point_positions
    from openptv2.algorithms.parameters import VolumePar
    from openptv2.algorithms.tracking_frame_buf import Frame, Target
    from openptv2.algorithms.trafo import metric_to_pixel
    from openptv2.correspondences import MatchedCoords
    from openptv2.storage import RunStore

    origin_dir = proptv_case_dir / "origin"
    files = sorted(origin_dir.glob("origin_*.txt"))
    if not files:
        raise FileNotFoundError(f"no origin_*.txt under {origin_dir}")

    rng = np.random.default_rng(seed)
    yaml_path, yaml_data, ptv, cpar = _prepare_scaffold(scaffold, out)
    res = out / "res"
    crit = yaml_data["criteria"]
    crit["eps0"] = eps0_mm
    vpar = VolumePar(
        X_lay=crit["X_lay"], Zmin_lay=crit["Zmin_lay"], Zmax_lay=crit["Zmax_lay"],
        cnx=crit["cnx"], cny=crit["cny"], cn=crit["cn"], csumg=crit["csumg"],
        corrmin=crit["corrmin"], eps0=eps0_mm,
    )

    cals_true = [
        Calibration.from_file(
            str(out / "cal" / f"cam{c + 1}.tif.ori"), str(out / "cal" / f"cam{c + 1}.tif.addpar")
        )
        for c in range(NUM_CAMS)
    ]
    # Reconstruction-side calibration is a fixed, small, per-camera
    # perturbation of the truth -- and IS what's written to cal/ and thus
    # what the tracker itself loads at runtime, matching a real experiment
    # (the "true" camera model is never actually known).
    cals_recon = [
        _perturb_calibration(c, rng, calib_angle_sigma_deg, calib_pos_sigma_mm, calib_cc_ppm)
        for c in cals_true
    ]
    for c in range(NUM_CAMS):
        cals_recon[c].write(
            str(out / "cal" / f"cam{c + 1}.tif.ori"), str(out / "cal" / f"cam{c + 1}.tif.addpar")
        )

    # Read every frame's truth up front: dropout must be a per-particle
    # streak across the whole sequence (see _streak_dropout_mask), which
    # needs the full particle count and frame count before the main loop.
    all_true_rows: list[list[tuple[int, float, float, float]]] = []
    for f in files:
        rows = []
        for line in f.read_text().strip().splitlines():
            if line.startswith("#"):
                continue
            parts = [float(p) for p in line.split()]
            pid = int(parts[0])
            x = (parts[1] - 0.5) * CUBE_SCALE
            y = (parts[2] - 0.5) * CUBE_SCALE
            z = (parts[3] - 0.5) * CUBE_SCALE
            rows.append((pid, x, y, z))
        all_true_rows.append(rows)
    n_particles = max(pid for rows in all_true_rows for pid, *_ in rows) + 1
    n_frames = len(files)
    dropout_masks = [
        _streak_dropout_mask(n_particles, n_frames, dropout_p, mean_streak_frames, rng)
        for _ in range(NUM_CAMS)
    ]

    store = RunStore.open(out, mode="a")
    n_true_total = n_matched_total = 0

    for i, true_rows in enumerate(all_true_rows):
        fn = FIRST + i
        n_true_total += len(true_rows)

        with open(res / f"origin_{fn}.txt", "w") as out_f:
            out_f.write("ID,X,Y,Z\n")
            for pid, x, y, z in true_rows:
                out_f.write(f"{pid},{x:.6f},{y:.6f},{z:.6f}\n")

        # 1-3: project via the TRUE calibration, then apply detection noise,
        # missed detections (streak-correlated, see dropout_masks above),
        # and image-merging per camera.
        per_cam_pix: list[np.ndarray] = []
        for c in range(NUM_CAMS):
            pts = []
            for pid, x, y, z in true_rows:
                if dropout_masks[c][pid, i]:
                    continue
                mx, my = img_coord((x, y, z), cals_true[c], cpar.mm)
                px, py = metric_to_pixel(mx, my, cpar)
                if not (0 <= px <= cpar.imx and 0 <= py <= cpar.imy):
                    continue
                px += rng.normal(0, noise_px)
                py += rng.normal(0, noise_px)
                pts.append((px, py))
            arr = np.array(pts, dtype=np.float64) if pts else np.zeros((0, 2))
            if len(arr) > 1:
                from scipy.spatial import cKDTree

                pairs = cKDTree(arr).query_pairs(r=merge_radius_px)
                drop = {b for _a, b in pairs}
                if drop:
                    arr = np.array([p for j, p in enumerate(arr) if j not in drop])
            per_cam_pix.append(arr)

        per_cam_targets = [
            [Target(pnr=j, x=p[0], y=p[1]) for j, p in enumerate(arr[np.argsort(arr[:, 1])])]
            for arr in per_cam_pix
        ]

        # 4: the real multi-camera correspondence solver.
        frm = Frame(num_cams=NUM_CAMS, max_targets=2000)
        for c in range(NUM_CAMS):
            frm.num_targets[c] = len(per_cam_targets[c])
            for j, t in enumerate(per_cam_targets[c]):
                frm.targets[c][j].pnr = t.pnr
                frm.targets[c][j].x = t.x
                frm.targets[c][j].y = t.y
                frm.targets[c][j].tnr = -1
                frm.targ_x[c][j] = t.x
                frm.targ_y[c][j] = t.y
                frm.targ_tnr[c][j] = -1

        corrected = [MatchedCoords(per_cam_targets[c], cpar, cals_recon[c]) for c in range(NUM_CAMS)]
        con, counts = alg_correspondences(
            frm, [mc._corrected for mc in corrected], vpar, cpar, cals_recon
        )
        total = counts[3] if len(counts) > 3 else sum(counts[:3])
        corresp_list = []
        for row in con[:total]:
            mapped = []
            for c in range(NUM_CAMS):
                idx = row.p[c]
                mapped.append(int(corrected[c]._corrected[idx].pnr) if idx >= 0 else -1)
            corresp_list.append(mapped)
        corresp = (
            np.array(corresp_list, dtype=np.int32) if corresp_list else np.zeros((0, NUM_CAMS), dtype=np.int32)
        )
        n_matched_total += len(corresp)

        # 5: triangulate with the (perturbed) reconstruction calibration.
        if len(corresp) > 0:
            flat = np.array(
                [corrected[c].get_by_pnrs(corresp[:, c]) for c in range(NUM_CAMS)]
            )
            pos, _ = alg_point_positions(flat.transpose(1, 0, 2), cpar, cals_recon, vpar)
        else:
            pos = np.zeros((0, 3))

        # Targets carry the tnr the real solver just assigned -- write them
        # (and the correspondences) from frm, not per_cam_targets.
        for c in range(NUM_CAMS):
            store.write_targets(c, fn, frm.targets[c][: frm.num_targets[c]])
        store.write_correspondences(frame=fn, pos_3d=pos, cam_target_ids=corresp)

    last = FIRST + len(files) - 1
    yaml_data["sequence"]["first"] = FIRST
    yaml_data["sequence"]["last"] = last
    yaml_path.write_text(yaml.safe_dump(yaml_data, sort_keys=False))

    print(
        f"wrote {len(files)} frames ({FIRST}-{last}) -> {out} (realistic pipeline): "
        f"{n_matched_total}/{n_true_total} real-correspondence matches "
        f"({100 * n_matched_total / n_true_total:.1f}%)"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case", choices=["500_25", "500_30"])
    ap.add_argument("--scaffold", default="test_data/synthetic_turbulent",
                     help="existing openptv2 dataset to clone cal/img/yaml from")
    ap.add_argument("--proptv-root", default=r"C:/Users/alex/Github/proPTV/data")
    ap.add_argument("--realistic", action="store_true",
                     help="simulate the real detection/correspondence error chain "
                          "instead of injecting ground-truth correspondences directly")
    ap.add_argument("--severity", choices=sorted(SEVERITY_PRESETS), default="moderate",
                     help="noise/dropout/merge/calibration operating point for --realistic "
                          "(eps0 is derived from noise_px, not part of the preset)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path("test_data") / f"proptv_{args.case}"
    if args.realistic:
        convert_realistic(
            Path(args.proptv_root) / args.case, Path(args.scaffold), out,
            seed=args.seed, **SEVERITY_PRESETS[args.severity],
        )
    else:
        convert(Path(args.proptv_root) / args.case, Path(args.scaffold), out)
