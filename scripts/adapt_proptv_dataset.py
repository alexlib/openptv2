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
import shutil
from pathlib import Path

import numpy as np
import yaml

FIRST = 10001
NUM_CAMS = 4
CUBE_SCALE = 40.0  # [0,1]^3 -> [-20,20]^3, inside the scaffold rig's FOV


def convert(proptv_case_dir: Path, scaffold: Path, out: Path) -> None:
    from openptv2.algorithms.calibration import Calibration
    from openptv2.algorithms.imgcoord import img_coord
    from openptv2.algorithms.parameters import ControlPar, MmNp
    from openptv2.algorithms.trafo import metric_to_pixel
    from openptv2.algorithms.tracking_frame_buf import Target
    from openptv2.storage import RunStore

    origin_dir = proptv_case_dir / "origin"
    files = sorted(origin_dir.glob("origin_*.txt"))
    if not files:
        raise FileNotFoundError(f"no origin_*.txt under {origin_dir}")

    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(scaffold, out)
    res = out / "res"
    img = out / "img"
    # Scaffold's own res/ and img/ are that dataset's leftover ascii/zarr
    # output (added.*, ptv_is.*, rt_is.*, run.zarr, camN.<frame>_targets,
    # ...) -- none of it applies to this dataset; wipe both clean rather
    # than pick individual globs.
    shutil.rmtree(res)
    res.mkdir()
    shutil.rmtree(img)
    img.mkdir()

    # Reuse the scaffold's own cal/ (.ori position/rotation, no distortion --
    # its .addpar files are already all-zero) as-is; only the multimedia
    # model is overridden below to plain air, since proPTV has no glass/water
    # -- this is a pure pinhole imaging model, nothing more.
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case", choices=["500_25", "500_30"])
    ap.add_argument("--scaffold", default="test_data/synthetic_turbulent",
                     help="existing openptv2 dataset to clone cal/img/yaml from")
    ap.add_argument("--proptv-root", default=r"C:/Users/alex/Github/proPTV/data")
    args = ap.parse_args()
    convert(
        Path(args.proptv_root) / args.case,
        Path(args.scaffold),
        Path("test_data") / f"proptv_{args.case}",
    )
