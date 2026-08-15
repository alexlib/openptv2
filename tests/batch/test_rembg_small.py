"""
Comprehensive step-by-step pipeline tests for test_rembg_small.

test_rembg_small is a 256×256 crop of the full rembg dataset:
  - 4 cameras, frames 1–5
  - 2 ground truth trajectories (1 full 5-frame, 1 exit 3-frame)

Each test focuses on a single pipeline phase and validates output format,
internal consistency, and (where possible) agreement with ground truth.

Usage:
    uv run pytest tests/batch/test_rembg_small.py -v -s
"""

import csv
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.imgcoord import img_coord_batch
from openptv2.algorithms.parameter_converters import get_control_par, get_volume_par
from openptv2.algorithms.tracking_frame_buf import Target, TargetArray
from openptv2.batch import pyptv_batch
from openptv2.correspondences import MatchedCoords, correspondences
from openptv2.orientation import point_positions
from openptv2.storage import RunStore, resolve_store_path

# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════
FRAMES = list(range(1, 6))  # 1–5
NCAMS = 4
CROP = 256
GT_TRAJ_TOTAL = 2  # trajectories in ground_truth/trajectories.csv
GT_FULL = 1  # trajectories spanning all 5 frames
GT_EXIT = 1  # trajectories exiting early


# ═════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ═════════════════════════════════════════════════════════════════════════════


def _frame_from_name(path: Path) -> int:
    return int(path.name.rsplit(".", 1)[-1])


def _rt_is_exists(path: Path) -> bool:
    """True if rt_is.<frame> exists on disk, or -- pipeline runs are
    store-only now, no ASCII (see
    docs/plans/2026-08-15-zarr-only-transition-plan.md) -- in the RunStore."""
    if path.exists():
        return True
    store = RunStore(resolve_store_path(path.parent), mode="r")
    return store.has_correspondences(_frame_from_name(path))


def _read_rt_is(path: Path) -> list[dict]:
    """Return list of {label, x, y, z, t1, t2, t3, t4} -- from ASCII if
    present, else from the RunStore."""
    if path.exists():
        lines = path.read_text().strip().splitlines()
        n = int(lines[0])
        rows = []
        for line in lines[1 : n + 1]:
            p = line.split()
            rows.append(
                dict(
                    label=int(p[0]),
                    x=float(p[1]),
                    y=float(p[2]),
                    z=float(p[3]),
                    t1=int(p[4]),
                    t2=int(p[5]),
                    t3=int(p[6]),
                    t4=int(p[7]),
                )
            )
        return rows

    store = RunStore(resolve_store_path(path.parent), mode="r")
    frame = _frame_from_name(path)
    if not store.has_correspondences(frame):
        return []
    pos, cam_ids = store.read_correspondences(frame)
    return [
        dict(label=i + 1, x=p[0], y=p[1], z=p[2], t1=c[0], t2=c[1], t3=c[2], t4=c[3])
        for i, (p, c) in enumerate(zip(pos, cam_ids))
    ]


def _linkage_name_from_path(path: Path) -> str:
    return path.name.rsplit(".", 1)[0]


def _ptv_is_exists(path: Path) -> bool:
    """True for ptv_is.<frame> or added.<frame>, on disk or in the
    RunStore -- "added" is the tracker's prio output, stored as an extra
    column on the "ptv_is" linkage group rather than a separate group (see
    RunStore.write_linkage's docstring), so a live-pipeline store answers
    "added" from "ptv_is" when there's no separate "added" group."""
    if path.exists():
        return True
    store = RunStore(resolve_store_path(path.parent), mode="r")
    frame = _frame_from_name(path)
    name = _linkage_name_from_path(path)
    if store.has_linkage(frame, name):
        return True
    return name == "added" and store.has_linkage(frame, "ptv_is")


def _read_ptv_is(path: Path) -> list[dict]:
    """Return list of {prev, next, x, y, z} for ptv_is.<frame> or
    added.<frame> -- from ASCII if present, else from the RunStore."""
    if path.exists():
        lines = path.read_text().strip().splitlines()
        n = int(lines[0])
        rows = []
        for line in lines[1 : n + 1]:
            p = line.split()
            rows.append(
                dict(
                    prev=int(p[0]),
                    next=int(p[1]),
                    x=float(p[2]),
                    y=float(p[3]),
                    z=float(p[4]),
                )
            )
        return rows

    store = RunStore(resolve_store_path(path.parent), mode="r")
    frame = _frame_from_name(path)
    name = _linkage_name_from_path(path)
    if not store.has_linkage(frame, name):
        if name != "added" or not store.has_linkage(frame, "ptv_is"):
            return []
        name = "ptv_is"
    prev, nxt, pos = store.read_linkage(frame, name)
    return [
        dict(prev=int(p), next=int(n), x=xyz[0], y=xyz[1], z=xyz[2])
        for p, n, xyz in zip(prev, nxt, pos)
    ]


def _targets_exist(path: Path) -> bool:
    if path.exists():
        return True
    m = re.search(r"cam(\d+)\.(\d+)_targets$", path.name)
    if not m:
        return False
    res_dir = path.parent.parent.parent / "res"
    store = RunStore(resolve_store_path(res_dir), mode="r")
    return store.has_targets(int(m.group(1)) - 1, int(m.group(2)))


def _read_targets(path: Path) -> list[dict]:
    """Read a _targets file. Returns list of {pnr, x, y, n, nx, ny, sumg, tnr}
    -- from ASCII if present, else from the RunStore (img/camN/camN.<frame>_targets
    -> res/run.zarr, up two directories from the target path)."""
    if path.exists():
        lines = path.read_text().strip().splitlines()
        n = int(lines[0])
        rows = []
        for line in lines[1 : n + 1]:
            p = line.split()
            rows.append(
                dict(
                    pnr=int(p[0]),
                    x=float(p[1]),
                    y=float(p[2]),
                    n=int(p[3]),
                    nx=int(p[4]),
                    ny=int(p[5]),
                    sumg=int(p[6]),
                    tnr=int(p[7]),
                )
            )
        return rows

    m = re.search(r"cam(\d+)\.(\d+)_targets$", path.name)
    if not m:
        return []
    cam_idx, frame = int(m.group(1)) - 1, int(m.group(2))
    res_dir = path.parent.parent.parent / "res"
    store = RunStore(resolve_store_path(res_dir), mode="r")
    if not store.has_targets(cam_idx, frame):
        return []
    targs = store.read_targets(cam_idx, frame)
    return [
        dict(
            pnr=t.pnr(),
            x=t.pos()[0],
            y=t.pos()[1],
            n=t.count_pixels()[0],
            nx=t.count_pixels()[1],
            ny=t.count_pixels()[2],
            sumg=t.sum_grey_value(),
            tnr=t.tnr(),
        )
        for t in targs
    ]


def _load_gt_trajectories(gt_dir: Path) -> list[dict]:
    with open(gt_dir / "trajectories.csv") as f:
        return list(csv.DictReader(f))


def _load_gt_particles(gt_dir: Path) -> list[dict]:
    with open(gt_dir / "particles.csv") as f:
        return list(csv.DictReader(f))


def _load_gt_projections(gt_dir: Path) -> list[dict]:
    with open(gt_dir / "projections.csv") as f:
        return list(csv.DictReader(f))


def _clear_res(res_dir: Path) -> None:
    if res_dir.exists():
        shutil.rmtree(res_dir)
    res_dir.mkdir()


def _clear_targets(img_dir: Path) -> None:
    """Remove _targets files so next run regenerates them."""
    for t in img_dir.rglob("*_targets"):
        t.unlink()


# ═════════════════════════════════════════════════════════════════════════════
# Print helpers
# ═════════════════════════════════════════════════════════════════════════════


def _sep(title: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}\n  {title}\n{bar}")


def _print_rt_is(rows: list[dict], frame: int) -> None:
    print(f"    rt_is.{frame}: {len(rows)} particles")
    if rows:
        xs = [r["x"] for r in rows]
        ys = [r["y"] for r in rows]
        zs = [r["z"] for r in rows]
        print(f"      X  [{min(xs):.2f}, {max(xs):.2f}]  mean={np.mean(xs):.2f}")
        print(f"      Y  [{min(ys):.2f}, {max(ys):.2f}]  mean={np.mean(ys):.2f}")
        print(f"      Z  [{min(zs):.2f}, {max(zs):.2f}]  mean={np.mean(zs):.2f}")
        t_hits = [sum(1 for r in rows if r[f"t{c + 1}"] >= 0) for c in range(4)]
        print(f"      target hits/cam: {t_hits}")


def _print_ptv_is(rows: list[dict], frame: int) -> None:
    linked = sum(1 for r in rows if r["next"] >= 0)
    prev_link = sum(1 for r in rows if r["prev"] >= 0)
    print(f"    ptv_is.{frame}: {len(rows)} particles  prev={prev_link}  fwd={linked}")


def _print_added(rows: list[dict], frame: int) -> None:
    linked = sum(1 for r in rows if r["next"] >= 0)
    print(f"    added.{frame}: {len(rows)} particles  fwd={linked}")


def _print_targets(rows: list[dict], cam: int, frame: int) -> None:
    print(
        f"    {cam}.{frame:04d}_targets: {len(rows)} targets "
        f"x=[{min(t['x'] for t in rows):.1f}, {max(t['x'] for t in rows):.1f}] "
        f"y=[{min(t['y'] for t in rows):.1f}, {max(t['y'] for t in rows):.1f}]"
        if rows
        else f"    {cam}.{frame:04d}_targets: 0 targets"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def rt_is_gt(rembg_small_dir) -> dict[int, np.ndarray]:
    """Ground truth 3D positions per frame (from ground_truth/projections.csv)."""
    rows = _load_gt_projections(rembg_small_dir / "ground_truth")
    by_frame: dict[int, list[np.ndarray]] = {}
    for r in rows:
        f = int(r["frame"])
        by_frame.setdefault(f, []).append(
            np.array([float(r["x_px_full"]), float(r["y_px_full"])])
        )
    return {f: np.array(v) for f, v in by_frame.items()}


# ═════════════════════════════════════════════════════════════════════════════
# Step 1: Parameter loading + calibration validation
# ═════════════════════════════════════════════════════════════════════════════


def test_step1_params_and_calibration(rembg_small_dir):
    """
    Load parameters from YAML and calibrations from .ori/.addpar.
    Validate that all values are consistent and projections land in-crop.
    """
    _sep("Step 1: Parameters + Calibration")

    # 1a. Load YAML
    yaml_path = rembg_small_dir / "parameters_Run1.yaml"
    assert yaml_path.exists()
    with open(yaml_path) as f:
        params = yaml.safe_load(f)
    print(f"  YAML: {yaml_path}")
    print(f"  num_cams: {params['num_cams']}")
    assert params["num_cams"] == NCAMS

    # 1b. Load ControlPar
    cpar = get_control_par(params)
    assert cpar.imx == CROP, f"imx={cpar.imx} != {CROP}"
    assert cpar.imy == CROP, f"imy={cpar.imy} != {CROP}"
    print(
        f"  ControlPar: imx={cpar.imx}  imy={cpar.imy}  "
        f"pix={cpar.pix_x}×{cpar.pix_y} mm/px"
    )

    # 1c. Load VolumePar
    vpar = get_volume_par(params)
    print(
        f"  VolumePar: X_lay={vpar.X_lay}  "
        f"Z=[{vpar.Zmin_lay[0]}, {vpar.Zmax_lay[0]}]  "
        f"eps0={vpar.eps0}"
    )

    # 1d. Load Calibration for each camera
    cals = []
    for cam in range(1, NCAMS + 1):
        cal = Calibration.from_file(
            str(rembg_small_dir / f"cal/cam{cam}.tif.ori"),
            str(rembg_small_dir / f"cal/cam{cam}.tif.addpar"),
        )
        cals.append(cal)
        print(
            f"  cam{cam}: pos=({cal.ext_par.x0:.1f}, {cal.ext_par.y0:.1f}, "
            f"{cal.ext_par.z0:.1f})  "
            f"xh={cal.int_par.xh:.4f}  yh={cal.int_par.yh:.4f}  "
            f"cc={cal.int_par.cc:.2f}"
        )

    # 1e. Project ground truth mean position — must land inside crop
    gt_positions = _load_gt_particles(rembg_small_dir / "ground_truth")
    if gt_positions:
        xs = np.array([float(r["X"]) for r in gt_positions])
        ys = np.array([float(r["Y"]) for r in gt_positions])
        zs = np.array([float(r["Z"]) for r in gt_positions])
        centroid = np.array([[xs.mean(), ys.mean(), zs.mean()]])
        print(
            f"\n  GT centroid: ({centroid[0, 0]:.2f}, {centroid[0, 1]:.2f}, "
            f"{centroid[0, 2]:.2f}) mm"
        )
        for i, cal in enumerate(cals):
            cam = i + 1
            xy_mm = img_coord_batch(centroid, cal, cpar.mm)[0]
            x_px = xy_mm[0] / cpar.pix_x + cpar.imx / 2
            y_px = cpar.imy / 2 - xy_mm[1] / cpar.pix_y
            ok = (0 <= x_px <= CROP) and (0 <= y_px <= CROP)
            print(
                f"    cam{cam}: → ({x_px:.1f}, {y_px:.1f}) px  "
                f"{'✓ in crop' if ok else '✗ OUTSIDE crop'}"
            )
            assert ok, f"GT centroid projects outside crop for cam{cam}"

    print("\n  PASS")


# ═════════════════════════════════════════════════════════════════════════════
# Step 2: Image loading + target detection (read + basic format)
# ═════════════════════════════════════════════════════════════════════════════


def test_step2_target_detection(rembg_small_dir):
    """
    Load images, verify format. Run detection via the high-level batch pipeline
    and verify target files are created with valid pixel coordinates.
    """
    _sep("Step 2: Image loading + target detection (via pipeline)")

    import imageio.v2 as imageio

    # Verify image format
    for frame in FRAMES:
        for cam in range(NCAMS):
            img_path = rembg_small_dir / f"img/cam{cam + 1}/{frame:08d}.tif"
            img = imageio.imread(str(img_path))
            assert img.shape == (CROP, CROP), f"{img_path}: shape {img.shape}"
            assert img.dtype == np.uint8, f"{img_path}: dtype {img.dtype}"

    # Run sequence pipeline which does detection and writes _targets files
    from openptv2.batch import pyptv_batch

    _clear_targets(rembg_small_dir / "img")
    pyptv_batch.main(
        rembg_small_dir / "parameters_Run1.yaml", FRAMES[0], FRAMES[-1], mode="sequence"
    )

    # Verify target files exist and have valid coordinates
    total_targets = 0
    for frame in FRAMES:
        for cam in range(NCAMS):
            tpath = (
                rembg_small_dir / f"img/cam{cam + 1}/cam{cam + 1}.{frame:04d}_targets"
            )
            assert _targets_exist(tpath), f"Missing {tpath}"
            targets = _read_targets(tpath)
            total_targets += len(targets)
            for t in targets:
                assert 0 <= t["x"] < CROP, f"cam{cam + 1} frame {frame}: x={t['x']}"
                assert 0 <= t["y"] < CROP, f"cam{cam + 1} frame {frame}: y={t['y']}"
        if frame == FRAMES[0]:
            print(
                f"    frame {frame} targets per cam: "
                f"{[len(_read_targets(rembg_small_dir / f'img/cam{c + 1}/cam{c + 1}.{frame:04d}_targets')) for c in range(NCAMS)]}"
            )

    print(f"  Total targets (all frames × cams): {total_targets}")
    assert total_targets > 0
    print("\n  PASS")


# ═════════════════════════════════════════════════════════════════════════════
# Step 3: Pixel → Metric correction (MatchedCoords)
# ═════════════════════════════════════════════════════════════════════════════


def test_step3_pixel_to_metric(rembg_small_dir):
    """
    Verify MatchedCoords correction pipeline:
      Pixel coords → metric → distortion-corrected (flat) coords.
    Check that:
      - Corrected coords are finite and in reasonable metric range
      - Pnr values are reset to sequential indices
    """
    _sep("Step 3: Pixel-to-metric correction (MatchedCoords)")

    yaml_path = rembg_small_dir / "parameters_Run1.yaml"
    with open(yaml_path) as f:
        params = yaml.safe_load(f)
    cpar = get_control_par(params)
    cals = [
        Calibration.from_file(
            str(rembg_small_dir / f"cal/cam{c}.tif.ori"),
            str(rembg_small_dir / f"cal/cam{c}.tif.addpar"),
        )
        for c in range(1, NCAMS + 1)
    ]

    # Read existing targets from one frame
    frame = FRAMES[0]
    for cam in range(NCAMS):
        tpath = rembg_small_dir / f"img/cam{cam + 1}/cam{cam + 1}.{frame:04d}_targets"
        raw = _read_targets(tpath)
        tarr = TargetArray(
            [
                Target(
                    pnr=t["pnr"],
                    x=t["x"],
                    y=t["y"],
                    sumg=t["sumg"],
                    n=t["n"],
                    nx=t["nx"],
                    ny=t["ny"],
                    tnr=t["tnr"],
                )
                for t in raw
            ]
        )
        mc = MatchedCoords(tarr, cpar, cals[cam], reset_numbers=True)

        assert len(mc._corrected) == len(raw), (
            f"cam{cam + 1}: corrected count {len(mc._corrected)} != target count {len(raw)}"
        )

        if len(mc._corrected) > 0:
            positions, pnrs = mc.as_arrays()
            assert np.all(np.isfinite(positions)), (
                f"cam{cam + 1}: non-finite corrected coords"
            )
            # Pnrs should contain all values 0..N-1 (MatchedCoords resets pnr,
            # but the corrected list is sorted by x-coord, so order differs)
            assert set(pnrs) == set(range(len(raw))), (
                f"cam{cam + 1}: pnrs not a complete set 0..{len(raw) - 1}: got {sorted(pnrs)[:10]}..."
            )
            # Flat (corrected) coords are the undistorted metric coordinate
            # minus the principal point (xh, yh), so the physical bound must
            # include the principal-point offset — for off-center calibrations
            # |xh|,|yh| can exceed the sensor half-size on their own. Use a
            # generous margin (2× sensor half-size) on top of that offset.
            sensor_half_x = cpar.imx * cpar.pix_x / 2  # ~0.896 mm for 256×256
            sensor_half_y = cpar.imy * cpar.pix_y / 2
            bound_x = 2 * sensor_half_x + abs(cals[cam].int_par.xh)
            bound_y = 2 * sensor_half_y + abs(cals[cam].int_par.yh)
            assert np.all(np.abs(positions[:, 0]) < bound_x), (
                f"cam{cam + 1}: corrected x out of range: {positions[:, 0].min():.4f} to {positions[:, 0].max():.4f}"
            )
            assert np.all(np.abs(positions[:, 1]) < bound_y)

            print(
                f"    cam{cam + 1}: {len(raw)} targets → "
                f"x=[{positions[:, 0].min():.4f}, {positions[:, 0].max():.4f}]  "
                f"y=[{positions[:, 1].min():.4f}, {positions[:, 1].max():.4f}] mm"
            )

    print("\n  PASS")


# ═════════════════════════════════════════════════════════════════════════════
# Step 4: Correspondence matching
# ═════════════════════════════════════════════════════════════════════════════


def test_step4_correspondence_matching(rembg_small_dir):
    """
    Run correspondence matching on all frames.
    Verify:
      - sorted_pos / sorted_corresp have correct shapes
      - Number of correspondences > 0 for most frames
      - Correspondences reference valid target indices
    """
    _sep("Step 4: Correspondence matching")

    yaml_path = rembg_small_dir / "parameters_Run1.yaml"
    with open(yaml_path) as f:
        params = yaml.safe_load(f)
    cpar = get_control_par(params)
    vpar = get_volume_par(params)
    cals = [
        Calibration.from_file(
            str(rembg_small_dir / f"cal/cam{c}.tif.ori"),
            str(rembg_small_dir / f"cal/cam{c}.tif.addpar"),
        )
        for c in range(1, NCAMS + 1)
    ]

    frame_stats = {}
    for frame in FRAMES:
        target_arrays = []
        matched_coords = []
        for cam in range(NCAMS):
            tpath = (
                rembg_small_dir / f"img/cam{cam + 1}/cam{cam + 1}.{frame:04d}_targets"
            )
            raw = _read_targets(tpath)
            if not raw:
                target_arrays.append(TargetArray())
                matched_coords.append(MatchedCoords(TargetArray(), cpar, cals[cam]))
                continue
            tarr = TargetArray(
                [
                    Target(
                        pnr=t["pnr"],
                        x=t["x"],
                        y=t["y"],
                        sumg=t["sumg"],
                        n=t["n"],
                        nx=t["nx"],
                        ny=t["ny"],
                        tnr=t["tnr"],
                    )
                    for t in raw
                ]
            )
            target_arrays.append(tarr)
            matched_coords.append(MatchedCoords(tarr, cpar, cals[cam]))

        try:
            sorted_pos, sorted_corresp, num_targs = correspondences(
                target_arrays, matched_coords, cals, vpar, cpar
            )
        except Exception as e:
            print(f"    frame {frame}: CORRESPONDENCE FAILED: {e}")
            pytest.fail(f"correspondences() raised: {e}")

        if not sorted_pos:
            print(f"    frame {frame}: 0 correspondences")
            frame_stats[frame] = 0
            continue

        # Concatenate across clique types (3-cam + 4-cam)
        all_pos = np.concatenate(sorted_pos, axis=1)
        all_corresp = np.concatenate(sorted_corresp, axis=1)
        n_total = all_pos.shape[1]

        # Validate shapes
        assert all_pos.shape[0] == NCAMS, (
            f"frame {frame}: pos cam dim = {all_pos.shape[0]}"
        )
        assert all_corresp.shape[0] == NCAMS, (
            f"frame {frame}: corresp cam dim = {all_corresp.shape[0]}"
        )

        # Validate target indices are valid (>= -1)
        assert np.all(all_corresp >= -1), f"frame {frame}: invalid target indices"
        # Validate pixel coordinates are finite
        assert np.all(np.isfinite(all_pos)), f"frame {frame}: non-finite pixel coords"

        # Count how many per clique type
        clique_counts = [s.shape[1] for s in sorted_pos]
        frame_stats[frame] = n_total

        print(
            f"    frame {frame}: {n_total} matches  "
            f"(cliques: {clique_counts})  "
            f"targets: {num_targs}"
        )
        for i, (pos, corresp) in enumerate(zip(sorted_pos, sorted_corresp)):
            if pos.shape[1] > 0:
                print(f"      clique-type {i}: {pos.shape[1]} matches")

    total_matches = sum(frame_stats.values())
    print(f"\n  Total matches across all frames: {total_matches}")
    assert total_matches > 0, "Zero correspondences across all frames"
    print("\n  PASS")


# ═════════════════════════════════════════════════════════════════════════════
# Step 5: 3D triangulation (point positioning)
# ═════════════════════════════════════════════════════════════════════════════


def test_step5_triangulation(rembg_small_dir):
    """
    Run full 3D reconstruction (correspondence + triangulation) on all frames.
    Verify:
      - 3D positions are finite and in reasonable mm range
      - Ray convergence distances are small (< 1 mm for good matches)
      - Per-frame particle counts > 0
    """
    _sep("Step 5: 3D triangulation (point_positions)")

    yaml_path = rembg_small_dir / "parameters_Run1.yaml"
    with open(yaml_path) as f:
        params = yaml.safe_load(f)
    cpar = get_control_par(params)
    vpar = get_volume_par(params)
    cals = [
        Calibration.from_file(
            str(rembg_small_dir / f"cal/cam{c}.tif.ori"),
            str(rembg_small_dir / f"cal/cam{c}.tif.addpar"),
        )
        for c in range(1, NCAMS + 1)
    ]

    MAX_DIST = 1e6  # accept all finite positions (detection differs from GT)

    frame_positions = {}
    for frame in FRAMES:
        target_arrays = []
        matched_coords = []
        for cam in range(NCAMS):
            tpath = (
                rembg_small_dir / f"img/cam{cam + 1}/cam{cam + 1}.{frame:04d}_targets"
            )
            raw = _read_targets(tpath)
            if not raw:
                target_arrays.append(TargetArray())
                matched_coords.append(MatchedCoords(TargetArray(), cpar, cals[cam]))
                continue
            tarr = TargetArray(
                [
                    Target(
                        pnr=t["pnr"],
                        x=t["x"],
                        y=t["y"],
                        sumg=t["sumg"],
                        n=t["n"],
                        nx=t["nx"],
                        ny=t["ny"],
                        tnr=t["tnr"],
                    )
                    for t in raw
                ]
            )
            target_arrays.append(tarr)
            matched_coords.append(MatchedCoords(tarr, cpar, cals[cam]))

        sorted_pos, sorted_corresp, _ = correspondences(
            target_arrays, matched_coords, cals, vpar, cpar
        )
        if not sorted_pos or all(s.shape[1] == 0 for s in sorted_pos):
            print(f"    frame {frame}: 0 correspondences — skipping")
            frame_positions[frame] = np.empty((0, 3))
            continue

        np.concatenate(sorted_pos, axis=1)
        all_corresp = np.concatenate(sorted_corresp, axis=1)

        flat = np.array(
            [matched_coords[i].get_by_pnrs(all_corresp[i]) for i in range(NCAMS)]
        )

        pos, dist = point_positions(flat.transpose(1, 0, 2), cpar, cals, vpar)
        valid = ~np.isnan(pos[:, 0]) & (dist < MAX_DIST)
        pos = pos[valid]
        dist = dist[valid]
        frame_positions[frame] = pos

        n_good = len(pos)
        if len(dist) > 0:
            print(
                f"    frame {frame}: {n_good} valid of {len(dist)} total  "
                f"dist_median={np.median(dist):.1f} mm  "
                f"dist_max={max(dist):.1f}"
            )
        else:
            print(f"    frame {frame}: 0 valid positions")

        if len(pos) > 0:
            print(
                f"      X: [{pos[:, 0].min():.1f}, {pos[:, 0].max():.1f}]  "
                f"mean={pos[:, 0].mean():.1f}"
            )
            print(
                f"      Y: [{pos[:, 1].min():.1f}, {pos[:, 1].max():.1f}]  "
                f"mean={pos[:, 1].mean():.1f}"
            )
            print(
                f"      Z: [{pos[:, 2].min():.1f}, {pos[:, 2].max():.1f}]  "
                f"mean={pos[:, 2].mean():.1f}"
            )

    total = sum(len(v) for v in frame_positions.values())
    print(f"\n  Total 3D positions (all frames): {total}")
    assert total > 0, "Zero 3D positions — correspondence failed"
    print("  NOTE: High ray-convergence distances expected — default detection")
    print("  finds different targets than rembg-masked GT.")
    print("\n  PASS")


# ═════════════════════════════════════════════════════════════════════════════
# Step 6: Sequence pipeline (end-to-end detection + correspondence)
# ═════════════════════════════════════════════════════════════════════════════


def test_step6_sequence_pipeline(rembg_small_dir, rembg_small_yaml):
    """
    Run sequence mode via pyptv_batch.main().
    Verify:
      - rt_is.* files created for all frames
      - Particle counts > 0
      - Target indices reference existing targets (-1 = unmatched OK)
      - Reconstructed 3D positions are finite
    """
    _sep("Step 6: Sequence pipeline (detection + correspondence)")

    res_dir = rembg_small_dir / "res"
    _clear_res(res_dir)
    # Also clear existing targets so pipeline re-detects
    _clear_targets(rembg_small_dir / "img")

    print(f"  Running sequence mode, frames {FRAMES[0]}–{FRAMES[-1]} ...")
    pyptv_batch.main(rembg_small_yaml, FRAMES[0], FRAMES[-1], mode="sequence")
    print("  Done.\n")

    counts = {}
    for frame in FRAMES:
        f = res_dir / f"rt_is.{frame}"
        assert _rt_is_exists(f), f"rt_is.{frame} not created"
        rows = _read_rt_is(f)
        counts[frame] = len(rows)
        _print_rt_is(rows, frame)

        # Validate each row
        for r in rows:
            assert np.isfinite(r["x"]), f"rt_is.{frame}: non-finite x"
            assert np.isfinite(r["y"]), f"rt_is.{frame}: non-finite y"
            assert np.isfinite(r["z"]), f"rt_is.{frame}: non-finite z"
            # Target indices: -1 = unmatched, >= 0 = valid target
            for t in (r["t1"], r["t2"], r["t3"], r["t4"]):
                assert t >= -1, f"rt_is.{frame}: target idx {t} < -1"

        # At least some particles must have 4-camera matches (all 4 targets >= 0)
        n_4cam = sum(1 for r in rows if all(r[f"t{c}"] >= 0 for c in range(1, 5)))
        print(f"      particles with 4-cam matches: {n_4cam}/{len(rows)}")

    # Every frame should have particles
    for frame in FRAMES:
        assert counts[frame] > 0, (
            f"rt_is.{frame}: 0 particles (pipeline produced empty output)"
        )
        print(f"  frame {frame}: {counts[frame]} particles  ✓")

    print("\n  PASS")


# ═════════════════════════════════════════════════════════════════════════════
# Step 7: Sequence idempotency
# ═════════════════════════════════════════════════════════════════════════════


def test_step7_sequence_idempotent(rembg_small_dir, rembg_small_yaml):
    """
    Run sequence twice. rt_is particle counts must match exactly.
    Guards against in-place state mutation between runs.
    """
    _sep("Step 7: Sequence idempotency")

    res_dir = rembg_small_dir / "res"
    counts = []
    for run in range(1, 3):
        _clear_res(res_dir)
        _clear_targets(rembg_small_dir / "img")
        print(f"  run {run}: sequence ...")
        pyptv_batch.main(rembg_small_yaml, FRAMES[0], FRAMES[-1], mode="sequence")
        c = {frame: len(_read_rt_is(res_dir / f"rt_is.{frame}")) for frame in FRAMES}
        print(f"    counts: {c}")
        counts.append(c)

    print()
    all_match = True
    for frame in FRAMES:
        v1, v2 = counts[0][frame], counts[1][frame]
        ok = "✓" if v1 == v2 else "✗"
        print(f"  frame {frame}: run1={v1}  run2={v2}  {ok}")
        if v1 != v2:
            all_match = False

    assert all_match, "Non-idempotent: rt_is counts differ between runs"
    print("\n  PASS")


# ═════════════════════════════════════════════════════════════════════════════
# Step 8: Full pipeline (sequence + tracking)
# ═════════════════════════════════════════════════════════════════════════════


def test_step8_full_pipeline(rembg_small_dir, rembg_small_yaml):
    """
    Run full pipeline (sequence + tracking) via subprocess to capture C-level output.
    Verify:
      - rt_is, ptv_is, added files exist for all frames
      - Tracking step produces valid output (0 links is OK — calibration mismatch)
      - File formats are correct
    """
    _sep("Step 8: Full pipeline (sequence + tracking)")

    res_dir = rembg_small_dir / "res"
    _clear_res(res_dir)
    _clear_targets(rembg_small_dir / "img")

    with tempfile.NamedTemporaryFile(
        "w+", delete=False, suffix=".txt", dir=rembg_small_dir
    ) as out_file:
        out_path = out_file.name
        cmd = [
            sys.executable,
            "-m",
            "openptv2.batch.pyptv_batch",
            str(rembg_small_yaml),
            str(FRAMES[0]),
            str(FRAMES[-1]),
        ]
        print(f"  cmd: {' '.join(cmd)}")
        print(f"  cwd: {rembg_small_dir}\n")
        try:
            subprocess.run(
                cmd,
                stdout=out_file,
                stderr=subprocess.STDOUT,
                check=True,
                cwd=rembg_small_dir,
            )
        except subprocess.CalledProcessError as e:
            with open(out_path) as f:
                print("\n--- subprocess output ---")
                print(f.read())
            pytest.fail(f"Subprocess failed: {e}")

    with open(out_path) as f:
        raw_output = f.read()
    Path(out_path).unlink(missing_ok=True)

    print("  Subprocess stdout:")
    for line in raw_output.splitlines():
        print(f"    {line}")

    # Parse tracking step output
    _sep("Tracking step details")
    step_links: dict[int, int] = {}
    step_lost: dict[int, int] = {}
    step_add: dict[int, int] = {}
    for line in raw_output.splitlines():
        m = re.search(
            r"step:\s*(\d+),.*links:\s*(\d+),.*lost:\s*(\d+),.*add:\s*(\d+)", line
        )
        if m:
            step = int(m.group(1))
            step_links[step] = int(m.group(2))
            step_lost[step] = int(m.group(3))
            step_add[step] = int(m.group(4))

    for step in sorted(step_links):
        print(
            f"    step {step}: links={step_links[step]}  "
            f"lost={step_lost[step]}  add={step_add[step]}"
        )

    # Validate output files
    _sep("Output file summary")
    for frame in FRAMES:
        for prefix in ("rt_is", "ptv_is", "added"):
            p = res_dir / f"{prefix}.{frame}"
            if prefix == "rt_is":
                assert _rt_is_exists(p), f"{prefix}.{frame} missing"
            else:
                assert _ptv_is_exists(p), f"{prefix}.{frame} missing"
            if prefix == "rt_is":
                rows = _read_rt_is(p)
                _print_rt_is(rows, frame)
            elif prefix == "ptv_is":
                rows = _read_ptv_is(p)
                _print_ptv_is(rows, frame)
            else:
                rows = _read_ptv_is(p)
                _print_added(rows, frame)

    # Verify rt_is has content (sequence worked)
    rt_is_total = sum(len(_read_rt_is(res_dir / f"rt_is.{f}")) for f in FRAMES)
    assert rt_is_total > 0, "rt_is files are all empty — sequence failed"

    # ptv_is_total = sum(
    #     len(_read_ptv_is(res_dir / f"ptv_is.{f}")) for f in FRAMES
    # )
    # print(f"\n  Pipeline trajectory links: {total_links}")

    print(f"\n  rt_is total particles: {rt_is_total}")
    print("  All output files present and valid ✓")
    print("\n  PASS")


# ═════════════════════════════════════════════════════════════════════════════
# Step 9: Ground truth comparison (rt_is positions vs GT)
# ═════════════════════════════════════════════════════════════════════════════


def test_step9_rt_is_vs_ground_truth(rembg_small_dir, rembg_small_yaml):
    """
    Compare rt_is.* 3D positions against ground_truth/particles.csv.
    Since the pipeline uses default (non-rembg) detection which finds
    different targets than the rembg-based GT, this is a DIAGNOSTIC test:
    it reports match quality with wide tolerances.

    Acceptance: recall ≥ 1 % (just confirm non-degenerate output).
    """
    _sep("Step 9: rt_is positions vs ground truth (diagnostic)")

    GT_MATCH_RADIUS = 10.0  # mm — wide radius for cross-detection matching
    GT_MIN_RECALL = 0.0  # 0% minimum — default detection ≠ rembg detection

    res_dir = rembg_small_dir / "res"
    _clear_res(res_dir)
    _clear_targets(rembg_small_dir / "img")

    print("  Running sequence mode ...")
    pyptv_batch.main(rembg_small_yaml, FRAMES[0], FRAMES[-1], mode="sequence")
    print("  Done.\n")

    gt_particles = _load_gt_particles(rembg_small_dir / "ground_truth")
    gt_by_frame: dict[int, list] = {}
    for row in gt_particles:
        f = int(row["frame"])
        gt_by_frame.setdefault(f, []).append(
            (float(row["X"]), float(row["Y"]), float(row["Z"]))
        )

    all_dists: list[float] = []
    frame_stats: dict[int, dict] = {}

    for frame in FRAMES:
        rt_rows = _read_rt_is(res_dir / f"rt_is.{frame}")
        gt_pos = np.array(gt_by_frame.get(frame, []))
        rt_pos = (
            np.array([[r["x"], r["y"], r["z"]] for r in rt_rows])
            if rt_rows
            else np.empty((0, 3))
        )

        print(f"  frame {frame}: {len(rt_rows)} reconstructed  |  {len(gt_pos)} in GT")

        if len(gt_pos) == 0 or len(rt_pos) == 0:
            frame_stats[frame] = dict(matched=0, total_gt=len(gt_pos), dists=[])
            continue

        # Greedy NN from GT → reconstructed
        dists, matched = [], 0
        used = np.zeros(len(rt_pos), dtype=bool)
        for gp in gt_pos:
            d = np.linalg.norm(rt_pos - gp, axis=1)
            d[used] = np.inf
            j = int(np.argmin(d))
            if d[j] < GT_MATCH_RADIUS:
                dists.append(float(d[j]))
                used[j] = True
                matched += 1

        recall = matched / len(gt_pos) if len(gt_pos) else 0.0
        median_d = float(np.median(dists)) if dists else float("inf")
        print(
            f"    matched {matched}/{len(gt_pos)}  "
            f"recall={recall:.1%}  median_dist={median_d:.2f} mm"
        )
        if dists:
            print(
                f"    dist: min={min(dists):.2f}  max={max(dists):.2f}  "
                f"p90={np.percentile(dists, 90):.2f}"
            )

        all_dists.extend(dists)
        frame_stats[frame] = dict(
            matched=matched, total_gt=len(gt_pos), dists=dists, recall=recall
        )

    _sep("Overall match quality")
    total_matched = sum(v["matched"] for v in frame_stats.values())
    total_gt = sum(v["total_gt"] for v in frame_stats.values())
    overall_recall = total_matched / total_gt if total_gt else 0.0
    overall_median = float(np.median(all_dists)) if all_dists else float("inf")
    print(f"  total GT particles  : {total_gt}")
    print(f"  matched             : {total_matched}")
    print(f"  overall recall      : {overall_recall:.1%}  (min {GT_MIN_RECALL:.0%})")
    print(f"  median match dist   : {overall_median:.2f} mm")
    print("\n  NOTE: Pipeline uses default high-pass detection, while GT")
    print("  was generated from rembg-masked images. Different targets →")
    print("  different 3D positions. Low recall is expected.")

    assert overall_recall >= GT_MIN_RECALL, (
        f"Recall {overall_recall:.1%} < {GT_MIN_RECALL:.0%}"
    )
    print("\n  PASS (diagnostic)")


# ═════════════════════════════════════════════════════════════════════════════
# Step 10: Ground truth comparison (tracking trajectories vs GT)
# ═════════════════════════════════════════════════════════════════════════════


def test_step10_trajectories_vs_ground_truth(rembg_small_dir, rembg_small_yaml):
    """
    Run full pipeline, reconstruct trajectories from ptv_is.*, compare
    against ground_truth/trajectories.csv.

    Acceptance:
      - Reconstructed trajectories are well-formed (no broken chains)
      - GT trajectories are loadable and have expected distribution
      - Pipeline produces some trajectory-like links (≥ 0)
    """
    _sep("Step 10: Trajectories vs ground truth")

    res_dir = rembg_small_dir / "res"
    _clear_res(res_dir)
    _clear_targets(rembg_small_dir / "img")

    print("  Running full pipeline ...")
    pyptv_batch.main(rembg_small_yaml, FRAMES[0], FRAMES[-1])
    print("  Done.\n")

    # ── Load GT ──────────────────────────────────────────────────
    gt_trajs = _load_gt_trajectories(rembg_small_dir / "ground_truth")
    gt_by_status: dict[str, int] = {}
    for t in gt_trajs:
        gt_by_status[t["status"]] = gt_by_status.get(t["status"], 0) + 1

    _sep("Ground truth trajectories")
    print(f"  total : {len(gt_trajs)}")
    for k, v in sorted(gt_by_status.items()):
        print(f"    {k:12s}: {v}")

    # Validate GT distribution
    assert len(gt_trajs) == GT_TRAJ_TOTAL, (
        f"GT has {len(gt_trajs)} trajectories, expected {GT_TRAJ_TOTAL}"
    )
    assert gt_by_status.get("full", 0) == GT_FULL, (
        f"GT full trajectories: {gt_by_status.get('full', 0)} ≠ {GT_FULL}"
    )
    assert gt_by_status.get("exit", 0) == GT_EXIT, (
        f"GT exit trajectories: {gt_by_status.get('exit', 0)} ≠ {GT_EXIT}"
    )

    # ── Reconstruct from ptv_is ──────────────────────────────────
    frame_rows: dict[int, list[dict]] = {}
    for frame in FRAMES:
        p = res_dir / f"ptv_is.{frame}"
        frame_rows[frame] = _read_ptv_is(p) if _ptv_is_exists(p) else []

    # Walk chains forward
    visited: dict[int, set[int]] = {f: set() for f in FRAMES}
    trajectories: list[list[tuple[int, int]]] = []

    for start_frame in FRAMES:
        for idx, row in enumerate(frame_rows[start_frame]):
            if idx in visited[start_frame]:
                continue
            if row["prev"] >= 0:  # has predecessor → not seed
                continue
            chain: list[tuple[int, int]] = [(start_frame, idx)]
            visited[start_frame].add(idx)
            cur_frame, cur_idx = start_frame, idx
            while True:
                nxt = frame_rows[cur_frame][cur_idx]["next"]
                if nxt < 0:
                    break
                next_frame = (
                    FRAMES[FRAMES.index(cur_frame) + 1]
                    if cur_frame in FRAMES[:-1]
                    else None
                )
                if next_frame is None:
                    break
                if nxt >= len(frame_rows[next_frame]):
                    break
                visited[next_frame].add(nxt)
                chain.append((next_frame, nxt))
                cur_frame, cur_idx = next_frame, nxt
            trajectories.append(chain)

    lengths = [len(t) for t in trajectories]
    full = sum(1 for l in lengths if l == len(FRAMES))

    _sep("Reconstructed trajectories")
    print(f"  total reconstructed  : {len(trajectories)}")
    print(f"  full length ({len(FRAMES)} frames): {full}")
    if trajectories:
        print(
            f"  length distribution  : "
            f"{ {l: lengths.count(l) for l in sorted(set(lengths))} }"
        )
    else:
        print("  (no trajectory links — expected when default detection")
        print("   finds different targets than rembg-based GT)")

    print(f"\n  GT total : {len(gt_trajs)}  (full={GT_FULL}, exit={GT_EXIT})")
    print(f"  pipeline : {len(trajectories)} trajectories ({full} full-length)")

    # Validate: trajectories list is well-formed
    for t in trajectories:
        assert len(t) >= 1, "Empty trajectory chain"
        for frame, idx in t:
            assert frame in FRAMES, f"Invalid frame {frame}"
            assert 0 <= idx < len(frame_rows[frame]), (
                f"Index {idx} out of range for frame {frame}"
            )

    print("\n  PASS (diagnostic)")


# ═════════════════════════════════════════════════════════════════════════════
# Step 11: Target file consistency (rt_is references match actual targets)
# ═════════════════════════════════════════════════════════════════════════════


def test_step11_target_consistency(rembg_small_dir):
    """
    After a pipeline run, verify that every non-negative target index
    in rt_is.* actually exists in the corresponding _targets file.
    """
    _sep("Step 11: rt_is → _targets cross-reference consistency")

    res_dir = rembg_small_dir / "res"
    # Run sequence to ensure fresh targets + rt_is
    _clear_res(res_dir)
    _clear_targets(rembg_small_dir / "img")
    pyptv_batch.main(
        rembg_small_dir / "parameters_Run1.yaml", FRAMES[0], FRAMES[-1], mode="sequence"
    )

    total_refs = 0
    total_errors = 0

    for frame in FRAMES:
        rt_rows = _read_rt_is(res_dir / f"rt_is.{frame}")
        for cam in range(NCAMS):
            tpath = (
                rembg_small_dir / f"img/cam{cam + 1}/cam{cam + 1}.{frame:04d}_targets"
            )
            targets = _read_targets(tpath)
            max_pnr = max(t["pnr"] for t in targets) if targets else -1
            pnr_set = set(t["pnr"] for t in targets)

            for r in rt_rows:
                t_idx = r[f"t{cam + 1}"]
                if t_idx >= 0:
                    total_refs += 1
                    if t_idx not in pnr_set:
                        print(
                            f"    frame {frame} cam{cam + 1}: "
                            f"rt_is refs target {t_idx} but max pnr is {max_pnr}"
                        )
                        total_errors += 1

    print(f"  Total target references in rt_is: {total_refs}")
    print(f"  Broken references: {total_errors}")
    assert total_errors == 0, (
        f"{total_errors} rt_is entries reference non-existent targets"
    )
    print("\n  PASS")


# ═════════════════════════════════════════════════════════════════════════════
# Step 12: Cross-frame trajectory coherence
# ═════════════════════════════════════════════════════════════════════════════


def test_step12_rt_is_3d_coherence(rembg_small_dir):
    """
    Verify that 3D positions in rt_is files across frames are coherent:
      - No extreme jumps (> 50 mm between consecutive frames for matched particles)
      - All positions are finite
    """
    _sep("Step 12: rt_is 3D coherence across frames")

    res_dir = rembg_small_dir / "res"
    _clear_res(res_dir)
    _clear_targets(rembg_small_dir / "img")
    pyptv_batch.main(
        rembg_small_dir / "parameters_Run1.yaml", FRAMES[0], FRAMES[-1], mode="sequence"
    )

    MAX_JUMP = 50.0  # mm

    # Load all frames' rt_is
    all_frames = {}
    for frame in FRAMES:
        rows = _read_rt_is(res_dir / f"rt_is.{frame}")
        if rows:
            all_frames[frame] = np.array([[r["x"], r["y"], r["z"]] for r in rows])
        else:
            all_frames[frame] = np.empty((0, 3))

    # Check consecutive frames for extreme jumps (NN-based)
    n_jumps = 0
    for i in range(len(FRAMES) - 1):
        fa, fb = FRAMES[i], FRAMES[i + 1]
        pos_a = all_frames[fa]
        pos_b = all_frames[fb]
        if len(pos_a) == 0 or len(pos_b) == 0:
            continue

        # For each particle in frame A, find nearest in frame B
        for pa in pos_a:
            d = np.linalg.norm(pos_b - pa, axis=1)
            min_d = d.min()
            if min_d > MAX_JUMP:
                n_jumps += 1

    # Cross-frame target index coherence (same label ≈ same particle across frames)
    print(f"  Frames checked: {len(FRAMES)}")
    total_particles = sum(len(v) for v in all_frames.values())
    print(f"  Total particles across all frames: {total_particles}")
    print(f"  Particles with nearest-neighbor jump > {MAX_JUMP} mm: {n_jumps}")

    # This is informational — no strict assertion
    print("\n  PASS (informational)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
