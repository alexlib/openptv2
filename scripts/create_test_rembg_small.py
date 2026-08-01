#!/usr/bin/env python3
"""Generate test_data/test_rembg_small from test_data/test_rembg.

Usage (from repo root):
    uv run python test_data/create_test_rembg_small.py

What it does:
  Phase 0: Load YAML parameters + calibration.
  Phase 1: Run 3D reconstruction from existing _targets files to produce
           ground-truth 3D positions for frames 1-5.
  Phase 2: Compute 256×256 crop windows (centered on mean 3D position),
           filter particles inside all 4 views.
  Phase 3: Link trajectories, smooth, classify (full/entry/exit/transient).
  Phase 4: Crop images + calibration images, shift targets, update .ori,
           write parameters_Run1.yaml, res/ files, and ground_truth CSVs.
"""

import copy
import csv
import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import yaml

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.imgcoord import img_coord_batch
from openptv2.algorithms.parameter_converters import get_control_par, get_volume_par
from openptv2.algorithms.tracking_frame_buf import Target, TargetArray
from openptv2.correspondences import MatchedCoords, correspondences
from openptv2.orientation import point_positions

# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════
SRC = Path("test_data/test_rembg")
DST = Path("test_data/test_rembg_small")
YAML_PATH = SRC / "parameters_Run1.yaml"

FRAMES = list(range(1, 6))  # 1-5 (all 5 frames used for images + 3D)
ALL_FRAMES = list(range(1, 6))  # images exist for all 5
NCAMS = 4
CROP = 256
LINK_MM = 3.0  # max inter-frame 3D distance (mm) for trajectory linking
TGT_PX = 5.0  # max pixel distance to match particle projection to a target


# Image naming helpers — rembg uses per-camera subdirs with %08d.tif
def img_rel(cam, frame):
    return f"img/cam{cam}/{frame:08d}.tif"


def target_rel(cam, frame):
    return f"img/cam{cam}/cam{cam}.{frame:04d}_targets"


def img_path(src, cam, frame):
    return src / img_rel(cam, frame)


def target_path(src, cam, frame):
    return src / target_rel(cam, frame)


# ══════════════════════════════════════════════════════════════════════════════
# I/O helpers (same format as create_test_cavity_small.py)
# ══════════════════════════════════════════════════════════════════════════════


def load_rt_is(path):
    """Return list of (label, x, y, z, t1, t2, t3, t4)."""
    lines = Path(path).read_text().strip().splitlines()
    n = int(lines[0])
    out = []
    for line in lines[1 : n + 1]:
        p = line.split()
        out.append(
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
    return out


def load_targets(path):
    """Return list of (idx, x, y, sumg, nx, ny, npix, flag)."""
    lines = Path(path).read_text().strip().splitlines()
    n = int(lines[0])
    out = []
    for line in lines[1 : n + 1]:
        p = line.split()
        out.append(
            (
                int(p[0]),
                float(p[1]),
                float(p[2]),
                int(p[3]),
                int(p[4]),
                int(p[5]),
                int(p[6]),
                int(p[7]),
            )
        )
    return out


def write_rt_is(path, rows):
    """rows: (label, x, y, z, t1, t2, t3, t4)."""
    lines = [str(len(rows))]
    for r in rows:
        lines.append(
            f"{r[0]:4d}  {r[1]:12.6f}  {r[2]:12.6f}  {r[3]:12.6f}  "
            f"{r[4]:5d}  {r[5]:5d}  {r[6]:5d}  {r[7]:5d}"
        )
    Path(path).write_text("\n".join(lines) + "\n")


def write_ptv_is(path, rows):
    """rows: (prev_id, next_id, x, y, z)."""
    lines = [str(len(rows))]
    for r in rows:
        lines.append(f"{r[0]:4d}  {r[1]:4d}  {r[2]:12.6f}  {r[3]:12.6f}  {r[4]:12.6f}")
    Path(path).write_text("\n".join(lines) + "\n")


def write_targets(path, rows):
    """rows: (idx, x, y, sumg, nx, ny, npix, flag)."""
    lines = [str(len(rows))]
    for t in rows:
        lines.append(
            f"{t[0]:4d}  {t[1]:10.4f}  {t[2]:10.4f}  "
            f"{t[3]:6d}  {t[4]:4d}  {t[5]:4d}  {t[6]:6d}  {t[7]:4d}"
        )
    Path(path).write_text("\n".join(lines) + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Projection helper
# ══════════════════════════════════════════════════════════════════════════════


def to_pixels(positions, cal, cpar):
    """(N,3) mm → (N,2) full-image pixel coords."""
    xy_mm = img_coord_batch(np.asarray(positions, dtype=np.float64), cal, cpar.mm)
    x = xy_mm[:, 0] / cpar.pix_x + cpar.imx / 2
    y = cpar.imy / 2 - xy_mm[:, 1] / cpar.pix_y
    return np.stack([x, y], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# Trajectory helpers (same as cavity)
# ══════════════════════════════════════════════════════════════════════════════


def nn_match(pos_a, pos_b, threshold):
    """Greedy nearest-neighbour: (N,3),(M,3) → matches[i]=j or -1."""
    result = np.full(len(pos_a), -1, dtype=int)
    if len(pos_b) == 0:
        return result
    used = np.zeros(len(pos_b), dtype=bool)
    for i, a in enumerate(pos_a):
        d = np.linalg.norm(pos_b - a, axis=1)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] < threshold:
            result[i] = j
            used[j] = True
    return result


def smooth_traj(frame_pos):
    """Degree-2 poly per axis (or lower). frame_pos: {frame: [x,y,z] or None}."""
    valid = [(f, p) for f, p in frame_pos.items() if p is not None]
    if len(valid) < 2:
        return dict(frame_pos)
    t = np.array([f for f, _ in valid], dtype=float)
    pts = np.array([p for _, p in valid])
    deg = min(2, len(valid) - 1)
    polys = [np.polyfit(t, pts[:, i], deg) for i in range(3)]
    out = {}
    for f, p in frame_pos.items():
        if p is not None:
            out[f] = [float(np.polyval(polys[i], f)) for i in range(3)]
        else:
            out[f] = None
    return out


def classify(frame_pos):
    valid = [f for f in FRAMES if frame_pos.get(f) is not None]
    if not valid:
        return "empty"
    if valid[0] == FRAMES[0] and valid[-1] == FRAMES[-1]:
        return "full"
    if valid[0] == FRAMES[0]:
        return "exit"
    if valid[-1] == FRAMES[-1]:
        return "entry"
    return "transient"


def nearest_tgt(xc, yc, tgt_list, threshold):
    """Return new_idx of nearest target in crop coords, or -1."""
    if not tgt_list:
        return -1
    best_d, best_i = np.inf, -1
    for t in tgt_list:
        d = ((t[1] - xc) ** 2 + (t[2] - yc) ** 2) ** 0.5
        if d < best_d:
            best_d, best_i = d, t[0]
    return best_i if best_d < threshold else -1


# ══════════════════════════════════════════════════════════════════════════════
# Phase 0 — Load YAML parameters + calibration
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Phase 0: Load parameters + calibration")
print("=" * 60)

with open(YAML_PATH) as f:
    yaml_params = yaml.safe_load(f)

cpar = get_control_par(yaml_params)
vpar = get_volume_par(yaml_params)

cals = [
    Calibration.from_file(
        str(SRC / f"cal/cam{c}.tif.ori"),
        str(SRC / f"cal/cam{c}.tif.addpar"),
    )
    for c in range(1, NCAMS + 1)
]

print(f"  Image size: {cpar.imx} x {cpar.imy}")
print(f"  Pixel size: {cpar.pix_x} x {cpar.pix_y} mm/px")
print(f"  Frames: {FRAMES}")
print(f"  Num cameras: {NCAMS}")

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — 3D reconstruction from existing _targets files
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 1: 3D reconstruction from existing _targets files")
print("=" * 60)

frame_positions = {}  # {frame: ndarray[N, 3]}  all 3D particle positions
frame_corresp = {}  # {frame: ndarray[N, 4]}  target indices (pnr per cam)

for frame in FRAMES:
    target_arrays = []
    matched_coords = []
    for cam in range(NCAMS):
        tpath = target_path(SRC, cam + 1, frame)
        if not tpath.exists():
            print(f"  [WARN] {tpath} not found — skipping frame {frame}")
            target_arrays.append(TargetArray())
            matched_coords.append(MatchedCoords(TargetArray(), cpar, cals[cam]))
            continue

        raw = load_targets(tpath)
        tarr = TargetArray(
            [
                Target(
                    pnr=t[0],
                    x=t[1],
                    y=t[2],
                    sumg=t[3],
                    n=t[4],
                    nx=t[5],
                    ny=t[6],
                    tnr=t[7],
                )
                for t in raw
            ]
        )
        target_arrays.append(tarr)
        matched_coords.append(MatchedCoords(tarr, cpar, cals[cam], reset_numbers=True))

    # Correspondence matching
    try:
        sorted_pos, sorted_corresp, num_targs = correspondences(
            target_arrays, matched_coords, cals, vpar, cpar
        )
    except Exception as e:
        print(f"  [WARN] correspondence matching failed for frame {frame}: {e}")
        frame_positions[frame] = np.empty((0, 3), dtype=np.float64)
        frame_corresp[frame] = np.empty((0, 4), dtype=np.int32)
        continue

    if not sorted_pos or all(s.shape[1] == 0 for s in sorted_pos):
        print(f"  Frame {frame}: no correspondences found")
        frame_positions[frame] = np.empty((0, 3), dtype=np.float64)
        frame_corresp[frame] = np.empty((0, 4), dtype=np.int32)
        continue

    # Concatenate across clique types (3-cam + 4-cam)
    all_pos = np.concatenate(sorted_pos, axis=1)  # (NCAMS, N, 2) pixel
    all_corresp = np.concatenate(sorted_corresp, axis=1)  # (NCAMS, N)

    # Get corrected (metric flat) coordinates for triangulation
    flat = np.array(
        [matched_coords[i].get_by_pnrs(all_corresp[i]) for i in range(NCAMS)]
    )  # (NCAMS, N, 2)

    # 3D triangulation
    pos, dist = point_positions(flat.transpose(1, 0, 2), cpar, cals, vpar)

    # Filter out bad triangulations: NaN or high ray convergence distance
    # dist < 1.0 mm is a reasonable threshold for valid triangulation
    MAX_DIST = 1.0
    valid = ~np.isnan(pos[:, 0]) & (dist < MAX_DIST)
    pos = pos[valid]
    corresp = (
        all_corresp[:, valid]
        if valid.sum() > 0
        else np.empty((NCAMS, 0), dtype=np.int32)
    )

    frame_positions[frame] = pos
    frame_corresp[frame] = corresp.T  # (N, 4)

    n_total = sum(num_targs) if num_targs else 0
    print(
        f"  Frame {frame}: {len(pos)}/{n_total} 3D particles from {num_targs} targets"
    )

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Compute crop windows from centroid of all 3D data
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 2: Compute crop windows")
print("=" * 60)

all_positions = np.vstack(
    [frame_positions[f] for f in FRAMES if frame_positions[f].shape[0] > 0]
)
if len(all_positions) == 0:
    print("  ERROR: No 3D positions found. Cannot compute crop window.")
    print("  Using default centroid (25, -30, 0) mm as fallback.")
    centroid_3d = np.array([[25.0, -30.0, 0.0]])
else:
    centroid_3d = all_positions.mean(axis=0, keepdims=True)
    print(
        f"  Mean 3D position from {len(all_positions)} particles: "
        f"({centroid_3d[0, 0]:.1f}, {centroid_3d[0, 1]:.1f}, "
        f"{centroid_3d[0, 2]:.1f}) mm"
    )

crop_offsets = {}
for i, cal in enumerate(cals):
    cam = i + 1
    xy_mm = img_coord_batch(np.asarray(centroid_3d, dtype=np.float64), cal, cpar.mm)[0]
    cx = int(round(xy_mm[0] / cpar.pix_x + cpar.imx / 2))
    cy = int(round(cpar.imy / 2 - xy_mm[1] / cpar.pix_y))
    ox = max(0, min(cpar.imx - CROP, cx - CROP // 2))
    oy = max(0, min(cpar.imy - CROP, cy - CROP // 2))
    crop_offsets[cam] = (ox, oy)
    print(f"  cam{cam}: centroid projection → ({cx},{cy}) px   crop TL ({ox},{oy})")

# Filter particles: keep those projecting inside all 4 crop windows
frame_subset = {}
for frame in FRAMES:
    rows = frame_positions[frame]
    if rows.shape[0] == 0:
        frame_subset[frame] = []
        continue
    in_all = np.ones(rows.shape[0], dtype=bool)
    for i, cal in enumerate(cals):
        cam = i + 1
        ox, oy = crop_offsets[cam]
        pxy = to_pixels(rows, cal, cpar)
        in_all &= (
            (pxy[:, 0] >= ox)
            & (pxy[:, 0] < ox + CROP)
            & (pxy[:, 1] >= oy)
            & (pxy[:, 1] < oy + CROP)
        )
    filtered_indices = np.where(in_all)[0]
    frame_subset[frame] = [
        (int(idx), float(r[0]), float(r[1]), float(r[2]))
        for idx, r in zip(filtered_indices, rows[filtered_indices])
    ]
    print(
        f"  Frame {frame}: {rows.shape[0]} total → {len(frame_subset[frame])} in window"
    )

# ══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Link trajectories across frames, smooth, classify
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 3: Linking + smoothing trajectories")
print("=" * 60)

tid_counter = 0
frame_tid = {f: {} for f in FRAMES}  # frame -> {subset_row_idx: tid}
traj_pos = {}  # tid -> {frame: [x,y,z] or None}

# Seed from first frame
for k, row in enumerate(frame_subset[FRAMES[0]]):
    traj_pos[tid_counter] = {f: None for f in FRAMES}
    traj_pos[tid_counter][FRAMES[0]] = [row[1], row[2], row[3]]
    frame_tid[FRAMES[0]][k] = tid_counter
    tid_counter += 1

# Link forward frame by frame
for fi in range(len(FRAMES) - 1):
    fa, fb = FRAMES[fi], FRAMES[fi + 1]
    rows_a = frame_subset[fa]
    rows_b = frame_subset[fb]
    matched_b = set()

    if rows_a and rows_b:
        pos_a = np.array([[r[1], r[2], r[3]] for r in rows_a])
        pos_b = np.array([[r[1], r[2], r[3]] for r in rows_b])
        matches = nn_match(pos_a, pos_b, LINK_MM)
        for ka, kb in enumerate(matches):
            tid = frame_tid[fa].get(ka)
            if tid is None:
                continue
            if kb >= 0:
                traj_pos[tid][fb] = [rows_b[kb][1], rows_b[kb][2], rows_b[kb][3]]
                frame_tid[fb][kb] = tid
                matched_b.add(kb)

    # New entries in fb with no match from fa
    for kb, row in enumerate(rows_b):
        if kb not in matched_b:
            traj_pos[tid_counter] = {f: None for f in FRAMES}
            traj_pos[tid_counter][fb] = [row[1], row[2], row[3]]
            frame_tid[fb][kb] = tid_counter
            tid_counter += 1

smoothed = {tid: smooth_traj(traj_pos[tid]) for tid in traj_pos}
classes = {tid: classify(traj_pos[tid]) for tid in traj_pos}
class_count: dict[str, int] = {}
for v in classes.values():
    class_count[v] = class_count.get(v, 0) + 1
print(f"  {len(traj_pos)} trajectories: {class_count}")

# ══════════════════════════════════════════════════════════════════════════════
# Phase 4 — Output dirs, crop images, filter targets, update cal + params
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 4: Cropping images + targets, writing output")
print("=" * 60)

for d in ("cal", "img", "res", "ground_truth", "plugins"):
    (DST / d).mkdir(parents=True, exist_ok=True)
for cam in range(1, NCAMS + 1):
    (DST / f"img/cam{cam}").mkdir(parents=True, exist_ok=True)

filtered_tgts: dict[int, dict[int, list]] = {
    f: {c: [] for c in range(1, NCAMS + 1)} for f in FRAMES
}

for frame in ALL_FRAMES:
    for cam in range(1, NCAMS + 1):
        ox, oy = crop_offsets[cam]

        # Crop image
        img = imageio.imread(str(img_path(SRC, cam, frame)))
        crop = img[oy : oy + CROP, ox : ox + CROP]
        imageio.imwrite(str(img_path(DST, cam, frame)), crop)
        print(f"  {img_rel(cam, frame)}: {img.shape} → {crop.shape}")

        # Load and filter targets
        all_tgts = load_targets(target_path(SRC, cam, frame))
        new_idx = 0
        for t in all_tgts:
            if ox <= t[1] < ox + CROP and oy <= t[2] < oy + CROP:
                if frame in filtered_tgts:
                    filtered_tgts[frame][cam].append(
                        (new_idx, t[1] - ox, t[2] - oy, t[3], t[4], t[5], t[6], t[7])
                    )
                new_idx += 1

        tgt_out = filtered_tgts[frame][cam] if frame in filtered_tgts else []
        write_targets(target_path(DST, cam, frame), tgt_out)

# Update .ori — shift principal point for each camera's crop
for i, cal in enumerate(cals):
    cam = i + 1
    ox, oy = crop_offsets[cam]
    cal_out = copy.deepcopy(cal)
    cal_out.int_par.xh += (cpar.imx / 2 - (ox + CROP / 2)) * cpar.pix_x
    cal_out.int_par.yh -= (cpar.imy / 2 - (oy + CROP / 2)) * cpar.pix_y
    cal_out.to_file(
        str(DST / f"cal/cam{cam}.tif.ori"),
        str(DST / f"cal/cam{cam}.tif.addpar"),
    )
    print(f"  cal/cam{cam}.tif.ori: principal point shifted")

# Crop calibration images
for cam in range(1, NCAMS + 1):
    ox, oy = crop_offsets[cam]
    cal_img = imageio.imread(str(SRC / f"cal/cam{cam}.tif"))
    cal_crop = cal_img[oy : oy + CROP, ox : ox + CROP]
    imageio.imwrite(str(DST / f"cal/cam{cam}.tif"), cal_crop)
    print(
        f"  cal/cam{cam}.tif: cropped [{oy}:{oy + CROP}, "
        f"{ox}:{ox + CROP}] → {cal_crop.shape}"
    )

# Copy calibration target file
shutil.copy(
    str(SRC / "cal/vertical_target.txt"),
    str(DST / "cal/vertical_target.txt"),
)

# Copy plugins
for plugin_file in (SRC / "plugins").iterdir():
    shutil.copy(str(plugin_file), str(DST / "plugins" / plugin_file.name))

# Write parameters_Run1.yaml with patched imx/imy and paths
yaml_out = copy.deepcopy(yaml_params)
yaml_out["ptv"]["imx"] = CROP
yaml_out["ptv"]["imy"] = CROP

# Update image paths in ptv section
yaml_out["ptv"]["img_name"] = [
    str(img_rel(cam, FRAMES[0])) for cam in range(1, NCAMS + 1)
]
# Update sequence base names
yaml_out["sequence"]["base_name"] = [
    f"img/cam{cam}/%08d.tif" for cam in range(1, NCAMS + 1)
]
yaml_out["sequence"]["first"] = FRAMES[0]
yaml_out["sequence"]["last"] = FRAMES[-1]

# Update shaking params
yaml_out["shaking"]["shaking_first_frame"] = FRAMES[0]
yaml_out["shaking"]["shaking_last_frame"] = FRAMES[-1]
yaml_out["shaking"]["shaking_max_num_frames"] = len(FRAMES)

# Update man_ori coordinates — 4 corners of the 256×256 crop
man_ori_coords = {}
for cam in range(NCAMS):
    man_ori_coords[f"camera_{cam}"] = {
        "point_1": {"x": CROP / 2, "y": CROP / 2},
        "point_2": {"x": CROP / 2, "y": CROP / 4},
        "point_3": {"x": CROP / 4, "y": CROP / 2},
        "point_4": {"x": CROP / 4, "y": CROP / 4},
    }
yaml_out["man_ori_coordinates"] = man_ori_coords

# Update criteria to match smaller volume if needed
# (keep original for now; user can tune)

with open(DST / "parameters_Run1.yaml", "w") as f:
    yaml.dump(yaml_out, f, default_flow_style=False, sort_keys=False)
print("  parameters_Run1.yaml written with imx/imy=256")

# ══════════════════════════════════════════════════════════════════════════════
# Phase 5 — Write res/ files + ground_truth/ CSVs
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Phase 5: Writing res/ files + ground truth CSVs")
print("=" * 60)

# Ordered per-frame particle list (tid, smoothed x, y, z) sorted by tid
frame_rows: dict[int, list] = {f: [] for f in FRAMES}
for tid, fp in smoothed.items():
    for f in FRAMES:
        if fp[f] is not None:
            frame_rows[f].append((tid, fp[f][0], fp[f][1], fp[f][2]))
for f in FRAMES:
    frame_rows[f].sort(key=lambda r: r[0])

# tid → row-index within frame
tid_row_idx: dict[int, dict[int, int]] = {
    f: {r[0]: i for i, r in enumerate(frame_rows[f])} for f in FRAMES
}
fi_of = {f: i for i, f in enumerate(FRAMES)}

for frame in FRAMES:
    rows = frame_rows[frame]
    fi = fi_of[frame]
    prev_f = FRAMES[fi - 1] if fi > 0 else None
    next_f = FRAMES[fi + 1] if fi < len(FRAMES) - 1 else None

    # rt_is — match each particle to nearest target in each camera
    rt_rows = []
    for tid, x, y, z in rows:
        tidx = []
        pos = np.array([[x, y, z]])
        for ci, cal in enumerate(cals):
            cam = ci + 1
            ox, oy = crop_offsets[cam]
            pxy = to_pixels(pos, cal, cpar)[0]
            xc, yc = pxy[0] - ox, pxy[1] - oy
            tidx.append(nearest_tgt(xc, yc, filtered_tgts[frame][cam], TGT_PX))
        rt_rows.append((tid, x, y, z, tidx[0], tidx[1], tidx[2], tidx[3]))
    write_rt_is(DST / f"res/rt_is.{frame}", rt_rows)

    # ptv_is — forward/backward links by row index
    ptv_rows = []
    for tid, x, y, z in rows:
        prev_id = tid_row_idx[prev_f].get(tid, -1) if prev_f else -1
        next_id = tid_row_idx[next_f].get(tid, -2) if next_f else -2
        ptv_rows.append((prev_id, next_id, x, y, z))
    write_ptv_is(DST / f"res/ptv_is.{frame}", ptv_rows)

    # added — particles whose first valid frame is this one (not first frame)
    added_rows = []
    if frame != FRAMES[0]:
        for tid, x, y, z in rows:
            first = min(f for f in FRAMES if smoothed[tid][f] is not None)
            if first == frame:
                next_id = tid_row_idx[next_f].get(tid, -2) if next_f else -2
                added_rows.append((-1, next_id, x, y, z))
    write_ptv_is(DST / f"res/added.{frame}", added_rows)

    print(f"  Frame {frame}: {len(rt_rows)} particles")

# ── Ground truth CSVs ────────────────────────────────────────────────────
with open(DST / "ground_truth/particles.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["particle_id", "frame", "X", "Y", "Z", "dx", "dy", "dz", "status"])
    for tid in sorted(smoothed):
        fp = smoothed[tid]
        status = classes[tid]
        for fi, frame in enumerate(FRAMES):
            pos = fp[frame]
            if pos is None:
                continue
            nxt = fp[FRAMES[fi + 1]] if fi < len(FRAMES) - 1 else None
            if nxt:
                dx, dy, dz = nxt[0] - pos[0], nxt[1] - pos[1], nxt[2] - pos[2]
                w.writerow(
                    [
                        tid,
                        frame,
                        f"{pos[0]:.6f}",
                        f"{pos[1]:.6f}",
                        f"{pos[2]:.6f}",
                        f"{dx:.6f}",
                        f"{dy:.6f}",
                        f"{dz:.6f}",
                        status,
                    ]
                )
            else:
                w.writerow(
                    [
                        tid,
                        frame,
                        f"{pos[0]:.6f}",
                        f"{pos[1]:.6f}",
                        f"{pos[2]:.6f}",
                        "",
                        "",
                        "",
                        status,
                    ]
                )

with open(DST / "ground_truth/trajectories.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["particle_id", "first_frame", "last_frame", "n_frames", "status"])
    for tid in sorted(smoothed):
        fp = smoothed[tid]
        valid = [f for f in FRAMES if fp[f] is not None]
        w.writerow([tid, valid[0], valid[-1], len(valid), classes[tid]])

with open(DST / "ground_truth/projections.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(
        [
            "particle_id",
            "frame",
            "cam",
            "x_px_full",
            "y_px_full",
            "x_px_crop",
            "y_px_crop",
        ]
    )
    for tid in sorted(smoothed):
        for frame in FRAMES:
            pos = smoothed[tid][frame]
            if pos is None:
                continue
            arr = np.array([pos])
            for ci, cal in enumerate(cals):
                cam = ci + 1
                ox, oy = crop_offsets[cam]
                pxy = to_pixels(arr, cal, cpar)[0]
                w.writerow(
                    [
                        tid,
                        frame,
                        cam,
                        f"{pxy[0]:.3f}",
                        f"{pxy[1]:.3f}",
                        f"{pxy[0] - ox:.3f}",
                        f"{pxy[1] - oy:.3f}",
                    ]
                )

# ── Summary ──────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"Done → {DST}/")
print(f"  Crop offsets: { {c: crop_offsets[c] for c in range(1, NCAMS + 1)} }")
print(f"  Trajectories: {len(traj_pos)} total  {class_count}")
for frame in FRAMES:
    print(f"  frame {frame}: {len(frame_rows[frame])} particles")
