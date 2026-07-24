#!/usr/bin/env python
"""Helper CLI for the openptv-calibrate skill.

Onboards a user with an OpenPTV dataset through headless calibration. The actual
calibration math lives in the openptv2 package (openptv2.autocalibration); this
helper only adds dataset inspection, seed-point onboarding (including an
interactive mouse click-picker), and reporting.

Subcommands (all write results to files; stdout is short status only):
    inspect  <dataset> --output report.json
    render   <dataset> --output-dir <dir>
    pick     <dataset>                        # interactive: click 4 pts/camera
    seed     <dataset> --seed-json seeds.json
    run      <dataset> --output report.json [--dry-run]

Always invoke via `uv run` from within the openptv2 checkout (so the project
venv with the compiled algorithms is on the path):
    uv run python skills/openptv-calibrate/scripts/calib.py inspect <dataset> -h
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# --- dataset layout ---------------------------------------------------------

def _num_cams(base: Path) -> int:
    import yaml
    pref = base / "parameters_Run1.yaml"
    if pref.exists():
        try:
            y = yaml.safe_load(pref.read_text()) or {}
            num = y.get("num_cams") or y.get("ptv", {}).get("num_cams")
            if num is not None:
                return int(num)
        except Exception:
            pass
    cands = sorted(base.glob("parameters_*.yaml"))
    if cands:
        try:
            y = yaml.safe_load(cands[0].read_text()) or {}
            num = y.get("num_cams") or y.get("ptv", {}).get("num_cams")
            if num is not None:
                return int(num)
        except Exception:
            pass
    from openptv2.algorithms.parameters import ControlPar
    ptv_path = base / "parameters" / "ptv.par"
    if ptv_path.exists():
        return ControlPar.from_file(ptv_path).num_cams
    return 0


def _cam_files(base: Path, cam: int) -> dict:
    from openptv2.autocalibration import cam_files as _resolve_cam_files

    img, ori, addpar = _resolve_cam_files(base, cam)
    return {
        "cal_image": str(img),
        "cal_image_exists": img.exists(),
        "ori": ori.exists(),
        "addpar": addpar.exists(),
        "targets": Path(str(img) + "_targets").exists(),
    }


def cmd_inspect(args) -> int:
    base = Path(args.dataset).resolve()
    par = base / "parameters"
    report: dict = {"dataset": str(base), "problems": []}

    num_cams = _num_cams(base)
    report["num_cams"] = num_cams

    calblock = _calblock_path(base)
    report["calblock"] = str(calblock)
    report["calblock_exists"] = calblock.exists()
    if not calblock.exists():
        report["problems"].append(f"missing 3D calibration body {calblock}")

    import yaml
    has_sortgrid = (par / "sortgrid.par").exists()
    yaml_path = None
    pref = base / "parameters_Run1.yaml"
    if pref.exists():
        yaml_path = pref
    else:
        cands = sorted(base.glob("parameters_*.yaml"))
        if cands:
            yaml_path = cands[0]

    if yaml_path is not None:
        try:
            y = yaml.safe_load(yaml_path.read_text()) or {}
            if "sortgrid" in y:
                has_sortgrid = True
        except Exception:
            pass

    report["sortgrid_par"] = has_sortgrid

    has_seed = (par / "man_ori.par").exists() and (par / "man_ori.dat").exists()
    if yaml_path is not None:
        try:
            y = yaml.safe_load(yaml_path.read_text()) or {}
            if "man_ori" in y or "man_ori_coordinates" in y:
                has_seed = True
        except Exception:
            pass

    report["man_ori_par"] = (par / "man_ori.par").exists() or (yaml_path is not None)
    report["man_ori_dat"] = (par / "man_ori.dat").exists() or (yaml_path is not None)
    report["cameras"] = [_cam_files(base, c) for c in range(num_cams)]

    have_seed = has_seed
    have_targets = num_cams > 0 and all(c["targets"] for c in report["cameras"])
    have_init = num_cams > 0 and all(
        c["ori"] and c["addpar"] for c in report["cameras"]
    )
    report["has_seed"] = have_seed
    report["has_targets"] = have_targets
    report["has_initial_guess"] = have_init
    report["ready_headless"] = bool(
        num_cams and calblock.exists() and report["sortgrid_par"]
        and have_seed and have_targets and have_init
    )
    if not have_init:
        report["problems"].append(
            "no initial guess (camN.tif.ori/.addpar missing for at least one camera) "
            "-> external_calibration needs a starting interior-parameter guess "
            "(cc/xh/yh) even for a from-scratch calibration; use `init` to write a "
            "naive default, then verify convergence via `run --dry-run` overlays"
        )
    if not have_seed:
        report["problems"].append(
            "no manual-orientation seed (man_ori.par/.dat or YAML configuration) -> use `pick` (mouse) "
            "or `render` + `seed`"
        )
    if not have_targets:
        report["problems"].append(
            "detected targets (camN.tif_targets) missing -> run target detection first"
        )

    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"Success! Inspection written to: {args.output}")
    print(f"ready_headless={report['ready_headless']}  "
          f"problems={len(report['problems'])}")
    return 0


# --- shared helpers ---------------------------------------------------------

def _calblock_path(base: Path) -> Path:
    """Resolve calblock path: YAML fixp_name first, then legacy names."""
    import yaml
    for yp in sorted(base.glob("parameters_*.yaml")):
        try:
            y = yaml.safe_load(yp.read_text()) or {}
            name = (y.get("cal_ori") or {}).get("fixp_name")
            if name:
                return base / name
        except Exception:
            pass
    # legacy fallbacks
    for name in ["target_on_a_side.txt", "splitter_target.txt"]:
        p = base / "cal" / name
        if p.exists():
            return p
    return base / "cal" / "target_on_a_side.txt"  # canonical missing path for error messages

def _calblock_map(base: Path):
    """Return (ids, xyz) for the 3D calibration body."""
    import numpy as np
    data = np.loadtxt(_calblock_path(base), ndmin=2)
    return data[:, 0].astype(int), data[:, 1:4]


def _corner_ids(ids, xyz) -> list[int]:
    """Four well-spread corner IDs (min/max X × min/max Y)."""
    x, y = xyz[:, 0], xyz[:, 1]
    out: list[int] = []
    for sx in (x.min(), x.max()):
        for sy in (y.min(), y.max()):
            d = (x - sx) ** 2 + (y - sy) ** 2
            pid = int(ids[d.argmin()])
            if pid not in out:
                out.append(pid)
    return out[:4]


def _draw_body_map(ax, ids, xyz, highlight=None) -> None:
    """Body point IDs in openptv2 axes: X left→right, Y bottom→top."""
    ax.scatter(xyz[:, 0], xyz[:, 1], s=25, c="k")
    for pid, p in zip(ids, xyz):
        ax.annotate(str(pid), (p[0], p[1]), fontsize=6,
                    textcoords="offset points", xytext=(2, 2))
    if highlight is not None:
        hp = xyz[list(ids).index(highlight)]
        ax.scatter([hp[0]], [hp[1]], s=220, facecolors="none",
                   edgecolors="red", linewidths=2.0)
    ax.set_xlabel("X  (left → right)")
    ax.set_ylabel("Y  (bottom → top)")
    ax.set_aspect("equal")
    ax.set_title("3D body point IDs" if highlight is None
                 else f"click point ID {highlight}")


def _reproject_grid(cal, cpar, xyz):
    """Reproject every 3D body point to pixels through cal."""
    import numpy as np

    from openptv2.algorithms.imgcoord import img_coord
    from openptv2.algorithms.trafo import metric_to_pixel
    out = []
    for p in xyz:
        xp, yp = img_coord(p, cal, cpar.mm)
        out.append(metric_to_pixel(xp, yp, cpar.imx, cpar.imy,
                                   cpar.pix_x, cpar.pix_y, cpar.chfield))
    return np.array(out)


def _initial_guess(base: Path, cam: int, cpar, pick_ids, clicks, ids_all, xyz_all):
    """External orientation from the 4 seed clicks; return (cal, reprojected grid)."""
    import numpy as np

    from openptv2.algorithms.calibration import Calibration
    from openptv2.algorithms.orientation import external_calibration

    from openptv2.autocalibration import cam_files as _resolve_cam_files

    _, ori, addpar = _resolve_cam_files(base, cam)
    if ori.exists() and addpar.exists():
        cal = Calibration.from_file(str(ori), str(addpar))
    else:
        cal = Calibration()
        cal.set_primary_point([0.0, 0.0, 50.0])
        cal.glass_par.d = cpar.mm.d[0]
    idx = [list(ids_all).index(i) for i in pick_ids]
    xyz4 = xyz_all[idx]
    external_calibration(cal, np.asarray(xyz4, float), np.asarray(clicks, float), cpar)
    return cal, _reproject_grid(cal, cpar, xyz_all)


# --- naive initial guess (for datasets with no prior .ori/.addpar at all) ---

def cmd_init(args) -> int:
    """Write a naive default .ori/.addpar for every camera missing one.

    `external_calibration`'s exterior solve (raw_orient) needs a starting
    interior-parameter guess (cc/xh/yh, distortion) even though it recomputes
    the exterior pose from the 4 seed points -- there is no calibration path
    that needs zero prior information. This writes a generic guess: camera
    on the -Z axis at `--distance` mm looking at the origin (angles=0, i.e.
    identity rotation -- correct only if the rig's world frame already has
    +Z pointing from body to camera; if `run`'s external_calibration fails to
    converge, the seed is more likely the cause than this guess, but a wildly
    different rig geometry could need `--distance` adjusted), `cc` guessed
    from the sensor width (`imx * pix_x`, i.e. ~90deg horizontal FOV -- a
    common starting assumption, not a measurement), xh=yh=0, zero distortion.
    Never overwrites an existing .ori/.addpar (use `run` for that, which
    keeps *.autobck backups).
    """
    from openptv2.algorithms.calibration import Calibration
    from openptv2.autocalibration import cam_files, _find_yaml

    base = Path(args.dataset).resolve()
    num_cams = _num_cams(base)
    yaml_path = _find_yaml(base)
    if yaml_path is None:
        print("ERROR: no parameters_*.yaml found", file=sys.stderr)
        return 1
    import yaml
    y = yaml.safe_load(yaml_path.read_text())
    ptv = y["ptv"]
    cc_guess = float(ptv["imx"]) * float(ptv["pix_x"])

    written = []
    for cam in range(num_cams):
        _, ori, addpar = cam_files(base, cam)
        if ori.exists() and addpar.exists():
            continue
        cal = Calibration(
            pos=[0.0, 0.0, -args.distance],
            angs=[0.0, 0.0, 0.0],
            prim_point=[0.0, 0.0, cc_guess],
            rad_dist=[0.0, 0.0, 0.0],
            decent=[0.0, 0.0],
            affine=[1.0, 0.0],
            glass=[0.0, 0.0, 1.0],
        )
        ori.parent.mkdir(parents=True, exist_ok=True)
        cal.write(str(ori), str(addpar))
        written.append(f"cam{cam + 1}")

    if not written:
        print("Nothing to do: every camera already has .ori/.addpar.")
    else:
        print(f"Success! Wrote naive initial guess for: {', '.join(written)} "
              f"(cc={cc_guess:.2f}mm, distance={args.distance}mm). "
              "Verify convergence with `run --dry-run` before trusting it.")
    return 0


# --- render for seed picking ------------------------------------------------

def cmd_render(args) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = Path(args.dataset).resolve()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    num_cams = _num_cams(base)

    from openptv2.autocalibration import cam_files as _resolve_cam_files, _find_yaml

    split_views = None
    yaml_path = _find_yaml(base)
    if yaml_path is not None:
        import yaml as _yaml
        ptv_params = _yaml.safe_load(yaml_path.read_text())["ptv"]
        if ptv_params.get("splitter"):
            import imageio.v3 as iio
            from openptv2.gui.ptv import image_split
            img0_path, _, _ = _resolve_cam_files(base, 0)
            raw = iio.imread(img0_path)
            if raw.ndim > 2:
                from skimage.color import rgb2gray
                from skimage.util import img_as_ubyte
                raw = img_as_ubyte(rgb2gray(raw[:, :, :3]))
            split_views = image_split(raw, order=ptv_params.get("splitter_order") or [0, 1, 3, 2])

    written = []
    for cam in range(num_cams):
        fig, ax = plt.subplots(figsize=(9, 7.2))
        try:
            if split_views is not None:
                ax.imshow(split_views[cam], cmap="gray")
            else:
                import imageio.v3 as iio
                img_path, _, _ = _resolve_cam_files(base, cam)
                ax.imshow(iio.imread(img_path), cmap="gray")
        except Exception:
            ax.text(0.5, 0.5, f"cannot load cam{cam + 1}", ha="center")
        ax.set_title(f"cam{cam + 1}: read off pixel (x,y) of 4 known grid points")
        ax.grid(True, color="cyan", alpha=0.3, linewidth=0.5)
        fig.tight_layout()
        dest = outdir / f"cam{cam + 1}_grid.png"
        fig.savefig(dest, dpi=110)
        plt.close(fig)
        written.append(str(dest))

    calblock = _calblock_path(base)
    if calblock.exists():
        ids, xyz = _calblock_map(base)
        fig, ax = plt.subplots(figsize=(9, 7.2))
        _draw_body_map(ax, ids, xyz)
        ax.set_title("3D calibration body: point IDs (X →, Y ↑) — use in `pick`/`seed`")
        fig.tight_layout()
        dest = outdir / "body_ids.png"
        fig.savefig(dest, dpi=110)
        plt.close(fig)
        written.append(str(dest))

    manifest = outdir / "render_manifest.json"
    manifest.write_text(json.dumps({"images": written}, indent=2))
    print(f"Success! {len(written)} images in {outdir} (manifest: {manifest})")
    return 0


# --- seed writing (shared) --------------------------------------------------

def _write_seed(base: Path, seeds: dict, num_cams: int) -> None:
    """Write the manual-orientation seed.

    YAML is the source of truth (openptv2.autocalibration reads it first), so we
    update every parameters_*.yaml's man_ori.nr + man_ori_coordinates. The legacy
    man_ori.par/.dat are also written for backward compatibility (e.g. optv).
    """
    per_cam = []
    for cam in range(num_cams):
        pts = seeds.get(str(cam)) or seeds.get(cam)
        if not pts or len(pts) != 4:
            raise ValueError(f"need exactly 4 points for camera {cam}")
        per_cam.append([(int(pid), float(x), float(y)) for pid, x, y in pts])

    # legacy .par/.dat (backward compat)
    par = base / "parameters"
    if par.is_dir():
        par_lines = [str(pid) for cam in per_cam for pid, _, _ in cam]
        dat_lines = [f"{x:.6f} {y:.6f}" for cam in per_cam for _, x, y in cam]
        (par / "man_ori.par").write_text("\n".join(par_lines) + "\n")
        (par / "man_ori.dat").write_text("\n".join(dat_lines) + "\n")

    # YAML (source of truth)
    import yaml
    for yp in sorted(base.glob("parameters_*.yaml")):
        y = yaml.safe_load(yp.read_text()) or {}
        nr = [pid for cam in per_cam for pid, _, _ in cam]
        y.setdefault("man_ori", {})["nr"] = nr
        coords = {}
        for cam_idx, cam in enumerate(per_cam):
            coords[f"camera_{cam_idx}"] = {
                f"point_{k + 1}": {"x": x, "y": yv}
                for k, (_, x, yv) in enumerate(cam)
            }
        y["man_ori_coordinates"] = coords
        yp.write_text(yaml.safe_dump(y, sort_keys=False))


def cmd_seed(args) -> int:
    """Write man_ori.par + man_ori.dat from a JSON file.

    seeds.json: {"0": [[id, x, y], x4], "1": [...], ...} for each camera.
    """
    base = Path(args.dataset).resolve()
    seeds = json.loads(Path(args.seed_json).read_text())
    try:
        _write_seed(base, seeds, _num_cams(base))
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print("Success! Wrote man_ori seed to YAML (+ man_ori.par/.dat for compat).")
    return 0


# --- interactive mouse click-picker -----------------------------------------

def cmd_pick(args) -> int:
    """Guided mouse click-picker for the manual-orientation seed.

    For each camera the script tells the user *which* point ID to click next
    (one at a time), highlighting it on a numbered map of the 3D body shown
    beside the cal image. After the 4 clicks it draws the initial-guess
    reprojection overlay so the user can confirm the seed before moving on.
    Writes man_ori.par + man_ori.dat. Falls back with guidance if no display.
    """
    import matplotlib
    for backend in ("TkAgg", "QtAgg", "MacOSX"):
        try:
            matplotlib.use(backend, force=True)
            break
        except Exception:
            continue
    import matplotlib.pyplot as plt

    if matplotlib.get_backend().lower() in ("agg", ""):
        print("ERROR: no interactive display available for clicking.\n"
              "Use `render` to save images, then `seed` with a JSON of points.",
              file=sys.stderr)
        return 2

    base = Path(args.dataset).resolve()
    num_cams = _num_cams(base)

    from openptv2.autocalibration import _find_yaml
    yaml_path = _find_yaml(base)
    if yaml_path is not None:
        import yaml as _yaml
        from openptv2.autocalibration import _cpar_from_ptv
        ptv_params = _yaml.safe_load(yaml_path.read_text())["ptv"]
        cpar = _cpar_from_ptv(ptv_params, num_cams)
    else:
        from openptv2.algorithms.parameters import ControlPar
        ptv_params = {}
        cpar = ControlPar.from_file(base / "parameters" / "ptv.par")
    splitter = bool(ptv_params.get("splitter"))
    splitter_order = ptv_params.get("splitter_order") or [0, 1, 3, 2]

    ids_all, xyz_all = _calblock_map(base)

    try:
        import imageio.v3 as iio
    except Exception:
        print("ERROR: imageio required to display images", file=sys.stderr)
        return 1

    if args.ids:
        pick_ids = [int(v) for v in args.ids.split(",")]
    else:
        pick_ids = _corner_ids(ids_all, xyz_all)
    if len(pick_ids) != 4:
        print("ERROR: need exactly 4 point IDs", file=sys.stderr)
        return 1
    missing = [i for i in pick_ids if i not in set(ids_all.tolist())]
    if missing:
        print(f"ERROR: point IDs not in calibration body: {missing}", file=sys.stderr)
        return 1
    print(f"Seed point IDs (click these, in order, on every camera): {pick_ids}")

    seeds: dict = {}
    from openptv2.autocalibration import cam_files as _resolve_cam_files

    split_views = None
    if splitter:
        # One shared raw frame -- split it once into per-camera quadrant
        # views so each camera's click step shows only its own view, not
        # the whole multiplexed frame.
        from openptv2.gui.ptv import image_split
        img0_path, _, _ = _resolve_cam_files(base, 0)
        raw = iio.imread(img0_path)
        if raw.ndim > 2:
            from skimage.color import rgb2gray
            from skimage.util import img_as_ubyte
            raw = img_as_ubyte(rgb2gray(raw[:, :, :3]))
        split_views = image_split(raw, order=splitter_order)

    for cam in range(num_cams):
        if split_views is not None:
            img = split_views[cam]
        else:
            img_path, _, _ = _resolve_cam_files(base, cam)
            img = iio.imread(img_path)
        fig, (axi, axb) = plt.subplots(1, 2, figsize=(15, 7),
                                       gridspec_kw={"width_ratios": [1.6, 1]})
        axi.imshow(img, cmap="gray")

        clicks = []
        for k, pid in enumerate(pick_ids):
            axb.clear()
            _draw_body_map(axb, ids_all, xyz_all, highlight=pid)
            axi.set_title(
                f"cam{cam + 1}/{num_cams}:  CLICK point ID {pid}   ({k + 1}/4)"
            )
            fig.canvas.draw_idle()
            print(f"cam{cam + 1}: click point ID {pid}  ({k + 1}/4) ...")
            c = plt.ginput(1, timeout=0)
            if not c:
                print(f"ERROR: no click for ID {pid} on cam{cam + 1}", file=sys.stderr)
                plt.close(fig)
                return 1
            clicks.append(c[0])
            axi.plot(c[0][0], c[0][1], "+", color="yellow", markersize=12, mew=2)
            axi.annotate(str(pid), c[0], color="yellow", fontsize=11,
                         textcoords="offset points", xytext=(6, 6))
            fig.canvas.draw_idle()

        # initial-guess overlay: external orientation from the 4 clicks
        try:
            _, rep = _initial_guess(base, cam, cpar, pick_ids, clicks, ids_all, xyz_all)
            axi.scatter(rep[:, 0], rep[:, 1], s=10, c="red", label="initial guess")
            axi.legend(loc="upper right", framealpha=0.7)
            axi.set_title(f"cam{cam + 1}: initial-guess overlay (red). "
                          f"Press any key to accept & continue")
        except Exception as e:  # seed too degenerate to orient
            axi.set_title(f"cam{cam + 1}: could not orient from seed ({e}). "
                          f"Press any key to continue")
        fig.canvas.draw_idle()
        plt.waitforbuttonpress()
        plt.close(fig)

        seeds[str(cam)] = [[pick_ids[i], clicks[i][0], clicks[i][1]] for i in range(4)]

    _write_seed(base, seeds, num_cams)
    print(f"\nSuccess! Wrote man_ori.par + man_ori.dat for {num_cams} cameras.")
    return 0


# --- run the calibration ----------------------------------------------------

def cmd_run(args) -> int:
    import numpy as np

    from openptv2.autocalibration import (
        _load_dataset_params,
        calibrate_dataset,
        cam_files,
        cross_camera_rcm,
        joint_plate_bundle_adjust,
        resolve_calblock,
    )

    results = calibrate_dataset(args.dataset, write=not args.dry_run, overlays=True)
    ok = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]
    report = {
        "dataset": str(Path(args.dataset).resolve()),
        "written": not args.dry_run,
        "mean_rms_px": float(np.mean([r.rms for r in ok])) if ok else None,
        "cameras": [
            {
                "cam": r.cam + 1,
                "matched": r.matched,
                "nfix": r.nfix,
                "rms_px": round(r.rms, 4) if r.error is None else None,
                "flags": r.flags,
                "error": r.error,
                "overlay": (
                    str(Path(args.dataset).resolve() / "cal" / "auto_calib"
                        / f"cam{r.cam + 1}_overlay.png")
                    if r.error is None else None
                ),
            }
            for r in results
        ],
    }

    base = Path(args.dataset).resolve()
    cpar = _load_dataset_params(base, resolve_calblock(base)).cpar
    rcm = cross_camera_rcm(results, cpar)
    report["cross_camera_rcm"] = rcm

    if args.joint_ba:
        ba_results, info = joint_plate_bundle_adjust(results, cpar)
        report["joint_ba"] = info
        before, after = info.get("rcm_before"), info.get("rcm_after")
        if "skipped" in info:
            print(f"joint BA: skipped ({info['skipped']})")
        elif before is not None and after is not None:
            print(f"joint BA: cross-camera RCM median {before:.3f} -> "
                  f"{after:.3f} mm")
            improved = after < before
            if improved and not args.dry_run:
                import shutil
                base_ = Path(args.dataset).resolve()
                for r in ba_results:
                    if r.error is not None or r.cal is None:
                        continue
                    _, ori, addpar = cam_files(base_, r.cam)
                    shutil.copy2(ori, Path(str(ori) + ".jointba-bck"))
                    shutil.copy2(addpar, Path(str(addpar) + ".jointba-bck"))
                    r.cal.write(str(ori).encode(), str(addpar).encode())
                print("  improved -> refined cals written")
            elif improved:
                print("  improved (dry-run, not written)")
            else:
                print("  did not improve -> cals unchanged")

    Path(args.output).write_text(json.dumps(report, indent=2))
    print(f"Success! Calibration report written to: {args.output}")
    for c in report["cameras"]:
        if c["error"]:
            print(f"  cam{c['cam']}: FAILED - {c['error']}")
        else:
            print(f"  cam{c['cam']}: matched {c['matched']}/{c['nfix']}  "
                  f"RMS={c['rms_px']}px  flags={'+'.join(c['flags'])}")
    if rcm is not None:
        print(f"cross-camera RCM ({rcm['n_common']} pts in all {cpar.num_cams} "
              f"cams, {rcm['n_points']} in >=2): "
              f"p50={rcm['median']:.3f}mm p95={rcm['p95']:.3f}mm "
              f"max={rcm['max']:.3f}mm")
        if rcm["p95"] > args.rcm_flag_mm:
            print("WARNING: cross-camera RCM high relative to per-camera RMS; "
                  "consider a "
                  "tracer self-calibration / dumbbell pass (see roadmap).")
    else:
        print("cross-camera RCM: n/a (need >=2 cameras and >=3 common points)")
    if report["mean_rms_px"] is not None:
        print(f"  mean RMS ({len(ok)}/{len(results)} cameras): "
              f"{report['mean_rms_px']:.3f}px  "
              f"({'written' if report['written'] else 'dry-run'})")
    else:
        print("  no camera converged")
    return 1 if failed else 0


def cmd_snapshot_refine(args) -> int:
    """Refine calibration using 3D particle positions from tracking results."""
    import copy

    import numpy as np
    import yaml

    from openptv2.algorithms.calibration import Calibration
    from openptv2.algorithms.orientation import full_calibration
    from openptv2.algorithms.tracking_frame_buf import read_targets
    from openptv2.autocalibration import (
        CANDIDATE_FLAGS,
        _find_yaml,
        _load_dataset_params,
        _reproject_px,
        rms_px,
    )

    base = Path(args.dataset).resolve()
    calblock = base / "cal" / "target_on_a_side.txt"
    dp = _load_dataset_params(base, calblock)
    cpar, num_cams = dp.cpar, dp.num_cams

    # sequence base names (strip %d so read_targets uses 4-digit padding)
    yaml_path = _find_yaml(base)
    if yaml_path is None:
        print("ERROR: no parameters YAML found", file=sys.stderr)
        return 1
    y = yaml.safe_load(yaml_path.read_text())
    seq_raw = y["sequence"]["base_name"]
    seq_bases = [str(base / s.replace("%d", "")) for s in seq_raw]

    # load current calibrations
    from openptv2.autocalibration import cam_files as _resolve_cam_files

    cals = []
    for cam in range(num_cams):
        _, ori, addpar = _resolve_cam_files(base, cam)
        cals.append(Calibration.from_file(str(ori), str(addpar)))

    # discover tracking result frames
    ptv_is_files = sorted(
        (base / "res").glob("ptv_is.*"),
        key=lambda p: int(p.suffix.lstrip(".")),
    )
    if not ptv_is_files:
        print("ERROR: no res/ptv_is.* files found", file=sys.stderr)
        return 1
    if args.frames:
        wanted = {int(f) for f in args.frames.split(",")}
        ptv_is_files = [p for p in ptv_is_files
                        if int(p.suffix.lstrip(".")) in wanted]

    print(f"Using {len(ptv_is_files)} frames from res/ptv_is.*")

    # accumulate per-camera correspondences
    per_cam_ref: list[list] = [[] for _ in range(num_cams)]
    per_cam_det: list[list] = [[] for _ in range(num_cams)]

    for pf in ptv_is_files:
        frame = int(pf.suffix.lstrip("."))
        lines = pf.read_text().splitlines()
        n = int(lines[0])
        pts3d = []
        for line in lines[1: n + 1]:
            parts = line.split()
            if len(parts) >= 5:
                pts3d.append([float(parts[2]), float(parts[3]), float(parts[4])])
        if not pts3d:
            continue
        pts3d_arr = np.array(pts3d)

        for cam in range(num_cams):
            targets = read_targets(seq_bases[cam], frame)
            if not targets:
                continue
            proj = np.array([_reproject_px(cals[cam], cpar.mm, p, cpar)
                             for p in pts3d_arr])
            tgt_xy = np.array([[t.x, t.y] for t in targets])
            for i, pp in enumerate(proj):
                d = np.linalg.norm(tgt_xy - pp, axis=1)
                j = int(np.argmin(d))
                if d[j] <= args.tol_px:
                    per_cam_ref[cam].append(pts3d_arr[i])
                    per_cam_det[cam].append(tgt_xy[j])

    # refine each camera
    for cam in range(num_cams):
        ref = np.array(per_cam_ref[cam]) if per_cam_ref[cam] else np.empty((0, 3))
        det = np.array(per_cam_det[cam]) if per_cam_det[cam] else np.empty((0, 2))

        if len(ref) < 6:
            print(f"  cam{cam + 1}: {len(ref)} correspondences — skipping (need ≥6)")
            continue

        rep_before = np.array([_reproject_px(cals[cam], cpar.mm, r, cpar) for r in ref])
        rms_before = rms_px(det, rep_before)

        # Conservative progression: start extrinsics-only (no flags).
        # Noisy tracking data makes intrinsic/distortion params ill-conditioned;
        # only accept a richer flag set if it genuinely reduces RMS and is NaN-free.
        # ponytail: no k3/p1/p2 — snapshot data is too noisy to constrain them
        SNAP_FLAGS = [
            [],                             # extrinsics only
            ["cc", "xh", "yh"],             # + principal point/distance
            ["cc", "xh", "yh", "k1"],       # + 1st-order radial
            ["cc", "xh", "yh", "k1", "k2"],  # + 2nd-order radial
        ]
        best_cal, best_rms, best_flags = None, rms_before, None
        for flags in SNAP_FLAGS:
            trial = copy.deepcopy(cals[cam])
            try:
                full_calibration(trial, ref, det, cpar, flags)
                rep = np.array([_reproject_px(trial, cpar.mm, r, cpar) for r in ref])
                r = rms_px(det, rep)
                if np.isnan(r) or np.any(np.isnan(rep)):
                    continue  # NaN → skip this flag set
                if r < best_rms * 0.99:  # require at least 1% improvement
                    best_rms, best_cal, best_flags = r, trial, flags
            except Exception:
                continue

        if best_cal is None:
            print(f"  cam{cam + 1}: {len(ref)} pts  before={rms_before:.3f}px  "
                  f"no improvement (NaN or RMS did not decrease)")
            continue

        print(f"  cam{cam + 1}: {len(ref)} pts  before={rms_before:.3f}px  "
              f"after={best_rms:.3f}px  flags={best_flags}")

        if not args.dry_run:
            cals[cam] = best_cal
            _, ori, addpar = _resolve_cam_files(base, cam)
            ori.rename(Path(str(ori) + ".snpbck"))
            addpar.rename(Path(str(addpar) + ".snpbck"))
            best_cal.write(str(ori), str(addpar))

    status = "dry-run" if args.dry_run else "written"
    print(f"Done ({status}).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="openptv-calibrate helper")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect")
    p.add_argument("dataset")
    p.add_argument("--output", required=True)
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("init")
    p.add_argument("dataset")
    p.add_argument("--distance", type=float, default=500.0,
                   help="naive camera-to-origin distance guess in mm (default 500)")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("render")
    p.add_argument("dataset")
    p.add_argument("--output-dir", required=True)
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("pick")
    p.add_argument("dataset")
    p.add_argument("--ids", help="optional 4 IDs (same for all cams), e.g. 2,3,71,73")
    p.set_defaults(func=cmd_pick)

    p = sub.add_parser("seed")
    p.add_argument("dataset")
    p.add_argument("--seed-json", required=True)
    p.set_defaults(func=cmd_seed)

    p = sub.add_parser("run")
    p.add_argument("dataset")
    p.add_argument("--output", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rcm-flag-mm", type=float, default=0.1,
                   help="warn if cross-camera RCM p95 exceeds this (mm, default 0.1)")
    p.add_argument("--joint-ba", action="store_true",
                   help="run joint plate bundle adjustment to lower cross-camera "
                        "RCM (writes refined cals only if it improves)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("snapshot-refine",
                       help="refine calibration from tracking result 3D positions")
    p.add_argument("dataset")
    p.add_argument("--tol-px", type=float, default=5.0,
                   help="match tolerance in pixels (default 5.0)")
    p.add_argument("--frames", default=None,
                   help="comma-separated frame numbers to use (default: all)")
    p.add_argument("--dry-run", action="store_true",
                   help="report without writing .ori/.addpar")
    p.set_defaults(func=cmd_snapshot_refine)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
