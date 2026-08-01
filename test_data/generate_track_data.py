#!/usr/bin/env python3
"""Generate missing test data for test_data/track/.

Creates img_orig/ (target files) and res_orig/ (correspondence results)
for frames 10095-10105 and 10240-10250, 2 cameras, 1920x1080 images.

IMPORTANT: tnr in target files must match particle index for
candidate sorting in trackcorr_loop_fast.

Run from repo root:
    uv run python test_data/generate_track_data.py
"""

import shutil
from pathlib import Path

import numpy as np

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.imgcoord import img_coord_batch
from openptv2.algorithms.parameters import ControlPar

TRACK_DIR = Path("test_data/track")
CAL_DIR = TRACK_DIR / "cal"
PARAM_DIR = TRACK_DIR / "parameters"
IMG_ORIG_DIR = TRACK_DIR / "img_orig"
RES_ORIG_DIR = TRACK_DIR / "res_orig"

FRAME_RANGES = [
    ("range1", 10095, 10105),
    ("range2", 10240, 10250),
]
NCAMS = 2
IMX = 1920
IMY = 1080


def write_targets(path, rows):
    lines = [str(len(rows))]
    for t in rows:
        lines.append(
            f"{t[0]:4d}  {t[1]:10.4f}  {t[2]:10.4f}  "
            f"{t[3]:6d}  {t[4]:4d}  {t[5]:4d}  {t[6]:6d}  {t[7]:4d}"
        )
    Path(path).write_text("\n".join(lines) + "\n")


def write_rt_is(path, rows):
    lines = [str(len(rows))]
    for r in rows:
        lines.append(
            f"{r[0]:4d}  {r[1]:12.6f}  {r[2]:12.6f}  {r[3]:12.6f}  "
            f"{r[4]:5d}  {r[5]:5d}  {r[6]:5d}  {r[7]:5d}"
        )
    Path(path).write_text("\n".join(lines) + "\n")


def write_ptv_is(path, rows):
    lines = [str(len(rows))]
    for r in rows:
        lines.append(f"{r[0]:4d}  {r[1]:4d}  {r[2]:12.6f}  {r[3]:12.6f}  {r[4]:12.6f}")
    Path(path).write_text("\n".join(lines) + "\n")


def write_added(path, rows):
    """Write added (prio) files with 6 columns: prev next x y z prio.

    Must match write_path_frame format in tracking_frame_buf.py line 486-488:
        f"{p.prev:4d} {p.next_idx:4d} {p.x[0]:10.3f} {p.x[1]:10.3f} {p.x[2]:10.3f} {p.prio:d}"
    read_path_frame line 371 reads prio_parts[5] (the 6th column) as the prio value.
    """
    lines = [str(len(rows))]
    for r in rows:
        # r = (prev, next, x, y, z, prio)
        lines.append(
            f"{r[0]:4d}  {r[1]:4d}  {r[2]:12.6f}  {r[3]:12.6f}  {r[4]:12.6f}  {r[5]:d}"
        )
    Path(path).write_text("\n".join(lines) + "\n")


def generate_trajectory(first_frame, last_frame, start_pos, velocity):
    frames = np.arange(first_frame, last_frame + 1, dtype=float)
    offset = frames - first_frame
    traj = np.column_stack(
        [
            start_pos[0] + velocity[0] * offset,
            start_pos[1] + velocity[1] * offset,
            start_pos[2] + velocity[2] * offset,
        ]
    )
    return dict(zip(map(int, frames), traj))


def _read_rt_is(path):
    if not path.exists():
        return []
    lines = path.read_text().strip().splitlines()
    if not lines:
        return []
    n = int(lines[0])
    rows = []
    for line in lines[1 : n + 1]:
        p = line.split()
        rows.append(
            (
                int(p[0]),
                float(p[1]),
                float(p[2]),
                float(p[3]),
                int(p[4]),
                int(p[5]),
                int(p[6]),
                int(p[7]),
            )
        )
    return rows


def main():
    print("Loading calibration and parameters...")
    cpar = ControlPar.from_file(str(PARAM_DIR / "ptv.par"))
    cals = [
        Calibration.from_file(
            str(CAL_DIR / f"cam{c}.tif.ori"),
            str(CAL_DIR / f"cam{c}.tif.addpar"),
        )
        for c in range(1, NCAMS + 1)
    ]

    np.random.seed(42)

    # 2 particles spanning all frames with small inter-frame motion
    particles = [
        ("p1", np.array([30.0, -5.0, 50.0]), np.array([0.3, 0.2, 0.1])),
        ("p2", np.array([80.0, 10.0, 30.0]), np.array([-0.2, -0.1, 0.2])),
    ]

    # Collect all frames across ranges
    all_frames = set()
    for _, first, last in FRAME_RANGES:
        all_frames.update(range(first, last + 1))
    all_frames = sorted(all_frames)
    absolute_first = min(all_frames)
    absolute_last = max(all_frames)

    trajectories = {}
    for pid, (name, start, vel) in enumerate(particles):
        trajectories[pid] = generate_trajectory(
            absolute_first, absolute_last, start, vel
        )

    # Clean and create directories
    if IMG_ORIG_DIR.exists():
        shutil.rmtree(IMG_ORIG_DIR)
    IMG_ORIG_DIR.mkdir()

    if RES_ORIG_DIR.exists():
        shutil.rmtree(RES_ORIG_DIR)
    RES_ORIG_DIR.mkdir()

    # -----------------------------------------------------------------------
    # PASS 1: Generate target files and rt_is correspondence files
    # -----------------------------------------------------------------------
    print("PASS 1: target and correspondence files...")
    for frame in all_frames:
        targets_by_cam = {c: [] for c in range(1, NCAMS + 1)}
        corres_rows = []

        for pid in range(len(particles)):
            traj = trajectories[pid]
            if frame not in traj:
                continue

            x, y, z = traj[frame]
            cam_target_pnrs = []

            for ci in range(NCAMS):
                cal = cals[ci]
                px_3d = np.array([[x, y, z]], dtype=np.float64)
                xy_mm = img_coord_batch(px_3d, cal, cpar.mm)[0]
                px = xy_mm[0] / cpar.pix_x + cpar.imx / 2
                py = cpar.imy / 2 - xy_mm[1] / cpar.pix_y

                px_f = px + np.random.uniform(-0.3, 0.3)
                py_f = py + np.random.uniform(-0.3, 0.3)

                if px_f < 0 or px_f >= IMX or py_f < 0 or py_f >= IMY:
                    cam_target_pnrs.append(-1)
                    continue

                # tnr MUST = pid for tracking to link across frames
                target = (pid, px_f, py_f, 50, 5, 5, 1000, pid)
                targets_by_cam[ci + 1].append(target)
                cam_target_pnrs.append(pid)

            while len(cam_target_pnrs) < 4:
                cam_target_pnrs.append(-1)

            visible = sum(1 for p in cam_target_pnrs[:NCAMS] if p >= 0)
            if visible >= 2:
                # Label = pid + 1 (1-indexed as in original C tests)
                corres_rows.append(
                    (
                        pid + 1,
                        x,
                        y,
                        z,
                        cam_target_pnrs[0],
                        cam_target_pnrs[1],
                        cam_target_pnrs[2],
                        cam_target_pnrs[3],
                    )
                )

        # Write target files
        for ci in range(NCAMS):
            cam = ci + 1
            write_targets(
                IMG_ORIG_DIR / f"cam{cam}.{frame:04d}_targets",
                targets_by_cam[cam],
            )

        # Write rt_is
        write_rt_is(RES_ORIG_DIR / f"rt_is.{frame}", corres_rows)

    # -----------------------------------------------------------------------
    # PASS 2: Generate linkage files (ptv_is, added)
    # -----------------------------------------------------------------------
    print("PASS 2: linkage files...")
    for frame in all_frames:
        corres_rows = _read_rt_is(RES_ORIG_DIR / f"rt_is.{frame}")
        prev_rows = (
            _read_rt_is(RES_ORIG_DIR / f"rt_is.{frame - 1}")
            if frame - 1 >= absolute_first
            else []
        )
        next_rows = (
            _read_rt_is(RES_ORIG_DIR / f"rt_is.{frame + 1}")
            if frame + 1 <= absolute_last
            else []
        )

        ptv_rows = []
        added_rows = []

        for k_row, row in enumerate(corres_rows):
            pid_label = row[0]

            prev_row_idx = -1
            next_row_idx = -1

            for pk, pr in enumerate(prev_rows):
                if pr[0] == pid_label:
                    prev_row_idx = pk
                    break

            for nk, nr in enumerate(next_rows):
                if nr[0] == pid_label:
                    next_row_idx = nk
                    break

            ptv_rows.append((prev_row_idx, next_row_idx, row[1], row[2], row[3]))
            if prev_row_idx < 0 and next_row_idx >= 0:
                # prio=4 for newly appearing particles (test expects p.prio == 4)
                added_rows.append((-1, next_row_idx, row[1], row[2], row[3], 4))

        write_ptv_is(RES_ORIG_DIR / f"ptv_is.{frame}", ptv_rows)
        write_added(RES_ORIG_DIR / f"added.{frame}", added_rows)

        sum(1 for _ in IMG_ORIG_DIR.glob(f"cam*.{frame:04d}_targets"))

    # Summary
    n_target_files = len(list(IMG_ORIG_DIR.glob("*_targets")))
    n_rt_files = len(list(RES_ORIG_DIR.glob("rt_is.*")))
    n_ptv_files = len(list(RES_ORIG_DIR.glob("ptv_is.*")))
    print(f"\nDone! Files generated in {IMG_ORIG_DIR}/ and {RES_ORIG_DIR}/")
    print(f"  Target files: {n_target_files}")
    print(f"  rt_is files:  {n_rt_files}")
    print(f"  ptv_is files: {n_ptv_files}")
    print(f"  added files:  {len(list(RES_ORIG_DIR.glob('added.*')))}")


if __name__ == "__main__":
    main()
