#!/usr/bin/env python3
"""Generate test_data/test_cavity_small from test_data/test_cavity.

Usage (from repo root):
    uv run python test_data/create_test_cavity_small.py
"""

import copy
import csv
import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from openptv2.algorithms.calibration import Calibration
from openptv2.algorithms.imgcoord import img_coord_batch
from openptv2.algorithms.parameters import ControlPar

# ── Config ────────────────────────────────────────────────────────────────────
SRC = Path("test_data/test_cavity")
DST = Path("test_data/test_cavity_small")
RES_SRC = SRC / "res_orig"  # C-reference tracking results (10001–10004)
IMG_SRC = SRC / "img"  # detected images + targets (all 5 frames)
ALL_FRAMES = list(range(10000, 10005))  # for images / targets
FRAMES = list(range(10001, 10005))  # for 3D ground truth (res_orig only)
NCAMS = 4
CROP = 256
CENTROID = np.array([[0.0, 2.5, 2.5]])
LINK_MM = 3.0  # max inter-frame 3D distance (mm) for trajectory linking
TGT_PX = 5.0  # max pixel distance to match particle projection to a target

# ── I/O ───────────────────────────────────────────────────────────────────────


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


# ── Projection ────────────────────────────────────────────────────────────────


def to_pixels(positions, cal, cpar):
    """(N,3) mm → (N,2) full-image pixel coords."""
    xy_mm = img_coord_batch(np.asarray(positions, dtype=np.float64), cal, cpar.mm)
    x = xy_mm[:, 0] / cpar.pix_x + cpar.imx / 2
    y = cpar.imy / 2 - xy_mm[:, 1] / cpar.pix_y
    return np.stack([x, y], axis=1)


# ── Trajectory helpers ────────────────────────────────────────────────────────


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
# Phase 1 — calibration, crop windows, filter particles
# ══════════════════════════════════════════════════════════════════════════════
print("Phase 1: calibration + crop windows")
cpar = ControlPar.from_file(SRC / "parameters/ptv.par")
cals = [
    Calibration.from_file(
        str(SRC / f"cal/cam{c}.tif.ori"),
        str(SRC / f"cal/cam{c}.tif.addpar"),
    )
    for c in range(1, NCAMS + 1)
]

crop_offsets = {}
for i, cal in enumerate(cals):
    cam = i + 1
    xy_mm = img_coord_batch(CENTROID, cal, cpar.mm)[0]
    cx = int(round(xy_mm[0] / cpar.pix_x + cpar.imx / 2))
    cy = int(round(cpar.imy / 2 - xy_mm[1] / cpar.pix_y))
    ox = max(0, min(cpar.imx - CROP, cx - CROP // 2))
    oy = max(0, min(cpar.imy - CROP, cy - CROP // 2))
    crop_offsets[cam] = (ox, oy)
    print(f"  cam{cam}: centroid → ({cx},{cy}) px   crop TL ({ox},{oy})")

frame_data = {f: load_rt_is(RES_SRC / f"rt_is.{f}") for f in FRAMES}
frame_subset = {}

for frame, rows in frame_data.items():
    if not rows:
        frame_subset[frame] = []
        continue
    pos = np.array([[r[1], r[2], r[3]] for r in rows])
    in_all = np.ones(len(rows), dtype=bool)
    for i, cal in enumerate(cals):
        cam = i + 1
        ox, oy = crop_offsets[cam]
        pxy = to_pixels(pos, cal, cpar)
        in_all &= (
            (pxy[:, 0] >= ox)
            & (pxy[:, 0] < ox + CROP)
            & (pxy[:, 1] >= oy)
            & (pxy[:, 1] < oy + CROP)
        )
    frame_subset[frame] = [rows[k] for k in np.where(in_all)[0]]
    print(f"  frame {frame}: {len(rows)} total → {len(frame_subset[frame])} in window")

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 — link trajectories across frames, smooth, classify
# ══════════════════════════════════════════════════════════════════════════════
print("Phase 2: linking + smoothing trajectories")

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
# Phase 3 — output dirs, crop images, filter targets, update .ori / ptv.par
# ══════════════════════════════════════════════════════════════════════════════
print("Phase 3: cropping images + targets")
for d in ("cal", "img", "img_orig", "parameters", "res", "ground_truth"):
    (DST / d).mkdir(parents=True, exist_ok=True)

# Crop images and build filtered target lists
# filtered_tgts only for FRAMES (3D ground truth frames); frame 10000 images
# are cropped but have no matching 3D data
filtered_tgts: dict[int, dict[int, list]] = {
    f: {c: [] for c in range(1, NCAMS + 1)} for f in FRAMES
}

for frame in ALL_FRAMES:
    for cam in range(1, NCAMS + 1):
        ox, oy = crop_offsets[cam]

        img_src = IMG_SRC / f"cam{cam}.{frame}"
        img = imageio.imread(str(img_src))
        crop = img[oy : oy + CROP, ox : ox + CROP]
        imageio.imwrite(str(DST / f"img/cam{cam}.{frame}"), crop)
        shutil.copy(
            str(DST / f"img/cam{cam}.{frame}"),
            str(DST / f"img_orig/cam{cam}.{frame}"),
        )

        all_tgts = load_targets(IMG_SRC / f"cam{cam}.{frame}_targets")
        new_idx = 0
        for t in all_tgts:
            if ox <= t[1] < ox + CROP and oy <= t[2] < oy + CROP:
                if frame in filtered_tgts:
                    filtered_tgts[frame][cam].append(
                        (new_idx, t[1] - ox, t[2] - oy, t[3], t[4], t[5], t[6], t[7])
                    )
                new_idx += 1

        tgt_out = filtered_tgts[frame][cam] if frame in filtered_tgts else []
        write_targets(DST / f"img/cam{cam}.{frame}_targets", tgt_out)
        shutil.copy(
            str(DST / f"img/cam{cam}.{frame}_targets"),
            str(DST / f"img_orig/cam{cam}.{frame}_targets"),
        )

# Update .ori — shift principal point for each camera's crop
for i, cal in enumerate(cals):
    cam = i + 1
    ox, oy = crop_offsets[cam]
    cal_out = copy.deepcopy(cal)
    # New image centre in full-image pixels: (ox + CROP/2, oy + CROP/2)
    # Principal point shifts by the offset from the original sensor centre
    cal_out.int_par.xh += (cpar.imx / 2 - (ox + CROP / 2)) * cpar.pix_x
    cal_out.int_par.yh -= (cpar.imy / 2 - (oy + CROP / 2)) * cpar.pix_y
    cal_out.to_file(
        str(DST / f"cal/cam{cam}.tif.ori"),
        str(DST / f"cal/cam{cam}.tif.addpar"),
    )

# Crop calibration images to the same window as the experiment images
for cam in range(1, NCAMS + 1):
    ox, oy = crop_offsets[cam]
    cal_img = imageio.imread(str(SRC / f"cal/cam{cam}.tif"))
    cal_crop = cal_img[oy : oy + CROP, ox : ox + CROP]
    imageio.imwrite(str(DST / f"cal/cam{cam}.tif"), cal_crop)
    print(
        f"  cal/cam{cam}.tif: cropped [{oy}:{oy + CROP}, {ox}:{ox + CROP}] → {cal_crop.shape}"
    )
shutil.copy(
    str(SRC / "cal/target_on_a_side.txt"), str(DST / "cal/target_on_a_side.txt")
)

# Copy parameters; patch only imaX / imaY in ptv.par
shutil.copytree(str(SRC / "parameters"), str(DST / "parameters"), dirs_exist_ok=True)
ptv_text = (DST / "parameters/ptv.par").read_text()
ptv_text = ptv_text.replace(f"{cpar.imx}\n", f"{CROP}\n", 1)
ptv_text = ptv_text.replace(f"{cpar.imy}\n", f"{CROP}\n", 1)
(DST / "parameters/ptv.par").write_text(ptv_text)

# ══════════════════════════════════════════════════════════════════════════════
# Phase 4 — write res/ files + ground_truth/ CSVs
# ══════════════════════════════════════════════════════════════════════════════
print("Phase 4: writing res/ files + ground truth CSVs")

# Frame 10000 has images + targets but no 3D tracking data (res_orig starts at 10001)
write_rt_is(DST / "res/rt_is.10000", [])
write_ptv_is(DST / "res/ptv_is.10000", [])
write_ptv_is(DST / "res/added.10000", [])

# Ordered per-frame particle list (tid, smoothed x, y, z) sorted by tid
frame_rows: dict[int, list] = {f: [] for f in FRAMES}
for tid, fp in smoothed.items():
    for f in FRAMES:
        if fp[f] is not None:
            frame_rows[f].append((tid, fp[f][0], fp[f][1], fp[f][2]))
for f in FRAMES:
    frame_rows[f].sort(key=lambda r: r[0])

# tid → row-index within frame (needed for ptv_is prev/next links)
tid_row_idx: dict[int, dict[int, int]] = {
    f: {r[0]: i for i, r in enumerate(frame_rows[f])} for f in FRAMES
}
fi_of = {f: i for i, f in enumerate(FRAMES)}

for frame in FRAMES:
    rows = frame_rows[frame]
    fi = fi_of[frame]
    prev_f = FRAMES[fi - 1] if fi > 0 else None
    next_f = FRAMES[fi + 1] if fi < len(FRAMES) - 1 else None

    # rt_is — match each particle projection to nearest target in each camera
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

    # added — particles whose first valid frame is this frame (and not frame 0)
    added_rows = []
    if frame != FRAMES[0]:
        for tid, x, y, z in rows:
            first = min(f for f in FRAMES if smoothed[tid][f] is not None)
            if first == frame:
                next_id = tid_row_idx[next_f].get(tid, -2) if next_f else -2
                added_rows.append((-1, next_id, x, y, z))
    write_ptv_is(DST / f"res/added.{frame}", added_rows)

# ── Ground truth CSVs ─────────────────────────────────────────────────────────
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

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nDone → {DST}/")
print(f"  Crop offsets: { {c: crop_offsets[c] for c in range(1, NCAMS + 1)} }")
print(f"  Trajectories: {len(traj_pos)} total  {class_count}")
for frame in FRAMES:
    print(f"  frame {frame}: {len(frame_rows[frame])} particles")
